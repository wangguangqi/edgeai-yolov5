import logging
import math
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from torchvision import models

import torch.utils.checkpoint as checkpoint
import sys
import os
from models.Models.SwinTransformer import SwinTransformerLayer
from functools import partial


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        
        self.layernorm2 = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, 4*c, bias=False)
        self.fc2 = nn.Linear(4*c, c, bias=False)

        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x1 = x
        x = self.layernorm1(x)
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x1   # <--- return two outputs, which is not kind to DDP

        x2 = x
        x = self.layernorm2(x)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x)) + x2

        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    # https://github.com/iscyy/yoloair
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, c2o, n=1, shortcut=True, g=1, e=[0.5,0.5], rate=[1.0 for _ in range(12)]):  # ch_in, ch_out, number, shortcut, groups, expansion 
        super().__init__()
        # c_ = int(c2 * e)  # hidden channels
        if isinstance(e,list):
            c1_ = int(c2o * e[0])
            c2_ = int(c2o * e[1])
        else:
            c1_ = int(c2o * e)
            c2_ = int(c2o * e)
        self.cv1 = Conv(c1, c1_, 1, 1)
        self.cv2 = Conv(c1, c2_, 1, 1)
        self.cv3 = Conv(c1_+c2_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c1_, c1_, shortcut, g, e=rate[i]) for i in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3STR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)


class CoT(nn.Module):
    # Contextual Transformer Networks https://arxiv.org/abs/2107.12292
    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)
        
        return k1+k2


class CoTBottleneck(nn.Module):
    # Standard bottleneck https://github.com/iscyy/yoloair
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(CoTBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = CoT(c_, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class CoT3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[CoTBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool= nn.AdaptiveAvgPool2d(1)
        self.max_pool= nn.AdaptiveMaxPool2d(1)

        self.fc1= nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False)
        self.relu1= nn.ReLU()
        self.fc2= nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        self.sigmoid= nn.Sigmoid()

    def forward(self,x):
        avg_out= self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out= self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out= avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention,self).__init__()
        self.conv1= nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)  # kernel size = 7 Padding is 3: (n - 7 + 1) + 2P = n 
        self.sigmoid= nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channelIn, channelOut):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channelIn)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4,pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos=pos_emb
        if self.pos :
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims ) // heads, 1, int(height)]), requires_grad=True)
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims )// heads, int(width), 1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
     
    def forward(self, x):
        n_batch, C, width, height = x.size() 
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        #print('q shape:{},k shape:{},v shape:{}'.format(q.shape,k.shape,v.shape))  #1,4,64,256
        content_content = torch.matmul(q.permute(0,1,3,2), k) #1,C,h*w,h*w
        # print("qkT=",content_content.shape)
        c1,c2,c3,c4=content_content.size()
        if self.pos:
       # print("old content_content shape",content_content.shape) #1,4,256,256
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(0,1,3,2)   #1,4,1024,64
           
            content_position = torch.matmul(content_position, q)# ([1, 4, 1024, 256])
            content_position=content_position if(content_content.shape==content_position.shape)else content_position[:,: , :c3,]
            assert(content_content.shape==content_position.shape)
        #print('new pos222-> shape:',content_position.shape)
       # print('new content222-> shape:',content_content.shape)
            energy = content_content + content_position
        else:
            energy=content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0,1,3,2)) #1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out

class BottleneckTransformer(nn.Module):
    # Transformer bottleneck
    #expansion = 1

    def __init__(self, c1, c2, stride=1, heads=4, mhsa=True, resolution=None,expansion=1):
        super(BottleneckTransformer, self).__init__()
        c_=int(c2*expansion)
        self.cv1 = Conv(c1, c_, 1,1)
        #self.bn1 = nn.BatchNorm2d(c2)
        if not mhsa:
            self.cv2 = Conv(c_,c2, 3, 1)
        else:
            self.cv2 = nn.ModuleList()
            self.cv2.append(MHSA(c2, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.cv2.append(nn.AvgPool2d(2, 2))
            self.cv2 = nn.Sequential(*self.cv2)
        self.shortcut = c1==c2 
        if stride != 1 or c1 != expansion*c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, expansion*c2, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion*c2)
            )
        self.fc1 = nn.Linear(c2, c2)     

    def forward(self, x):
        out=x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))
        return out
    

class BoT3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1,e=0.5,e2=1,w=20,h=20):  # ch_in, ch_out, number, , expansion,w,h
        super(BoT3, self).__init__()
        c_ = int(c2*e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[BottleneckTransformer(c_ ,c_, stride=1, heads=4,mhsa=True,resolution=(w,h),expansion=e2) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))      


class HorLayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).# https://ar5iv.labs.arxiv.org/html/2207.14284
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError # by iscyy/air
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        nn.init.trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = HorLayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = HorLayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x

class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )
        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        # B, C, H, W = x.shape gnconv [512]by iscyy/air
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]
        x = self.proj_out(x)

        return x

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)


class HorBlock(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = HorLayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)
        self.norm2 = HorLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape # [512]
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class HorNet(nn.Module):

    def __init__(self, index, in_chans, depths, dim_base, drop_path_rate=0.,layer_scale_init_value=1e-6, 
        gnconv=[
            partial(gnconv, order=2, s=1.0/3.0),
            partial(gnconv, order=3, s=1.0/3.0),
            partial(gnconv, order=4, s=1.0/3.0),
            partial(gnconv, order=5, s=1.0/3.0), # GlobalLocalFilter
        ],
    ):
        super().__init__()
        dims = [dim_base, dim_base * 2, dim_base * 4, dim_base * 8]
        self.index = index
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers by iscyy/air
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            HorLayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    HorLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks by iscyy/air
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        if not isinstance(gnconv, list):
            gnconv = [gnconv, gnconv, gnconv, gnconv]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[HorBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i] # by iscyy/air

        self.apply(self._init_weights)

    def _init_weights(self, m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.trunc_normal_(m.weight, std=.02)
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.downsample_layers[self.index](x)
        x = self.stages[self.index](x)
        return x
