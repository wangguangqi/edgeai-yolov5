# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel

# 把one-hot label转换为soft label
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    	普通的BCE损失非常依赖样本标记的正确率，如果某个样本标记错误，比如把正样本标记成负样本，那么会带来很大的误差。
    	而smoothBCE可以有效的减少由于样本标记错误带来的误差。如果看过SVM算法的话，感觉有点类似于硬间隔和软间隔
    	# 可以看这篇博客讲解label_smooth https://blog.csdn.net/racesu/article/details/107214035?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-107214035-blog-123999241.pc_relevant_default&spm=1001.2101.3001.4242.1&utm_relevant_index=2
    """
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    """
    	在focalloss里面，我们关注两个参数，alpha和gamma 一个用来控制正负样本的权重，一个用来控制易难样本的权重
    """
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        """
        	当使用的参数为 mean(在pytorch1.7.1中elementwise_mean已经弃用)会对N个样本的loss进行平均之后返回
        	当使用的参数为 sum会对N个样本的loss求和
        	表示直接返回n分样本的loss
        """ 
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, kpt_label=False):
        super(ComputeLoss, self).__init__()
        self.kpt_label = kpt_label
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        # Define criteria
        # 定义评价标准 cls代表类别的BCE loss obj的BCElos为判断第i个网格中的第j个box是否负责对应的object
        # 这里的pos_weight为对应的参数 在模型训练的yaml文件中可以调整

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCE_kptv = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        """
 		    对标签做平滑,eps=0就代表不做标签平滑,那么默认cp=1,cn=0
            后续对正类别赋值cp，负类别赋值cn           
        """
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # 这里进行标签平滑处理 cp代表positive的标签值 cn代表negative的标签值
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        # 获取fl_gamma参数，如果大于0的时候采用focalloss，否则使用BCE
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        """
            获取detect层
        """
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        """
            每一层预测值所占的权重比，分别代表浅层到深层，小特征到大特征，4.0对应着P3，1.0对应P4,0.4对应P5。
            如果是自己设置的输出不是3层，则返回[4.0, 1.0, 0.25, 0.06, .02]，可对应1-5个输出层P3-P7的情况。
        """
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        """
            autobalance 默认为 False，yolov5中目前也没有使用 ssi = 0即可
        """
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        """
            赋值各种参数,gr是用来设置IoU的值在objectness loss中做标签的系数, 
            使用代码如下：
		    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
            train.py源码中model.gr=1，也就是说完全使用标签框与预测框的CIoU值来作为该预测框的objectness标签。
        """
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'nkpt':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj, lkpt, lkptv = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89], device=device) / 10.0
        '''
        从build_targets函数中构建目标标签，获取标签中的tcls, tbox, indices, anchors
        tcls = [[cls1,cls2,...],[cls1,cls2,...],[cls1,cls2,...]]
        tcls.shape = [nl,N]
        tbox = [[[gx1,gy1,gw1,gh1],[gx2,gy2,gw2,gh2],...],
        
        indices = [[image indices1,anchor indices1,gridj1,gridi1],
        		   [image indices2,anchor indices2,gridj2,gridi2],
        		   ...]]
        anchors = [[aw1,ah1],[aw2,ah2],...]		  
        '''
        tcls, tbox, tkpt, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        '''
		p.shape = [nl,bs,na,nx,ny,no]
		nl 为 预测层数，一般为3
		na 为 每层预测层的anchor数，一般为3
		nx,ny 为 grid的w和h
		no 为 输出数，为5 + nc (5:x,y,w,h,obj,nc:分类数)
		'''

        for i, pi in enumerate(p):  # layer index, layer predictions
            '''
            a:所有anchor的索引
            b:标签所属image的索引
            gridy:标签所在grid的y，在0到ny-1之间
            gridy:标签所在grid的x，在0到nx-1之间
            '''
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            '''
            pi.shape = [bs,na,nx,ny,no]
            tobj.shape = [bs,na,nx,ny]
            '''
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                """
            	ps为batch中第b个图像第a个anchor的第gj行第gi列的output
            	ps.shape = [N,5+nc],N = a[0].shape,即符合anchor大小的所有标签数
            	"""
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                # 有目标的情况，我们对obj计算误差
                # bx = (2*σ(offsetX)−0.5)+gridX
                # by = (2*σ(offsetY)−0.5)+gridY   
                # xy的预测范围为-0.5~1.5
                # wh的预测范围是0~4倍anchor的w和h，             
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # 计算bw和bh 由于这是  pwh = (pwh.sigmoid() * 2) ** 2  的值域范围在-4~4之间
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                '''
                只有当CIOU=True时，才计算CIOU，否则默认为GIOU
                '''
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                if self.kpt_label:
                    #Direct kpt prediction
                    pkpt_x = ps[:, 6::3] * 2. - 0.5
                    pkpt_y = ps[:, 7::3] * 2. - 0.5
                    pkpt_score = ps[:, 8::3]
                    #mask
                    kpt_mask = (tkpt[i][:, 0::2] != 0)
                    lkptv += self.BCEcls(pkpt_score, kpt_mask.float()) 
                    #l2 distance based loss
                    #lkpt += (((pkpt-tkpt[i])*kpt_mask)**2).mean()  #Try to make this loss based on distance instead of ordinary difference
                    #oks based loss
                    d = (pkpt_x-tkpt[i][:,0::2])**2 + (pkpt_y-tkpt[i][:,1::2])**2
                    s = torch.prod(tbox[i][:,-2:], dim=1, keepdim=True)
                    kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0))/torch.sum(kpt_mask != 0)
                    # lkpt += kpt_loss_factor*((1 - torch.exp(-d/(s*(4*sigmas**2)+1e-9)))*kpt_mask).mean()
                    lkpt += kpt_loss_factor*((1 - torch.exp(-d/(2*(s*sigmas)**2+1e-9)))*kpt_mask).mean()
                # Objectness
                '''
                通过gr用来设置IoU的值在objectness loss中做标签的比重, 
                '''
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    '''
               		ps[:, 5:].shape = [N,nc],用 self.cn 来填充型为[N,nc]得Tensor。
               		self.cn通过smooth_BCE平滑标签得到的，使得负样本不再是0，而是0.5 * eps
                	'''
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    '''
                    self.cp 是通过smooth_BCE平滑标签得到的，使得正样本不再是1，而是1.0 - 0.5 * eps
                    '''
                    t[range(n), tcls[i]] = self.cp
                    '''
                    计算用sigmoid+BCE分类损失
                    '''
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
			
            """
			pi[..., 4]所存储的是预测的obj
			"""
            obji = self.BCEobj(pi[..., 4], tobj)
            '''
			self.balance[i]为第i层输出层所占的权重，在init函数中已介绍
			将每层的损失乘上权重计算得到obj损失
			'''
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # 根据超参数对各个Loss进行平衡
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkptv *= self.hyp['cls']
        lkpt *= self.hyp['kpt']
        bs = tobj.shape[0]  # batch size
        # hyp.yaml中设置了每种损失所占比重，分别对应相乘
        loss = lbox + lobj + lcls + lkpt + lkptv
        return loss * bs, torch.cat((lbox, lobj, lcls, lkpt, lkptv, loss)).detach()

    def build_targets(self, p, targets):
        """
            na = 3,表示每个预测层anchor的个数
            targets 为一个batch中所有的标签，包括标签所属的image,以及class,x,y,w,h,x1,y1,...,x17,y17
            targets = [[image1,class1,x,y,w,h,x1,y1,...,x17,y17],
                       [image2,class2,x,y,w,h,x1,y1,...,x17,y17],
                       ...
                       [imageN,classN,x,y,w,h,x1,y1,...,x17,y17]
                      ]
            nt为一个batch中所有标签的数量
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, tkpt, indices, anch = [], [], [], [], []
        """
            gain是为了最终将坐标所属grid坐标限制在坐标系中，不要超过范围
            其中41是为了对应：image, class, x, y, w, h, x1, y1, ...,x17,y17,ai
            x,y,w,h = nx,ny,nw,nh
            nx,ny作为当前输出层的grid大小

        """
        if self.kpt_label:
            gain = torch.ones(41, device=targets.device)  # normalized to gridspace gain
        else:
            gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        
        """
            ai.shape = [na,nt]
            ai = [[0,0,0,.....],
                [1,1,1,...],
                [2,2,2,...]]
            这么做的目的是为了给targets增加一个属性，即当前标签所属的anchor索引
        """
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        """
            targets.repeat(na, 1, 1).shape = [na,nt,6]
            ai[:, :, None].shape = [na,nt,1](None在list中的作用就是在插入维度1)
            ai[:, :, None] = [[[0],[0],[0],.....],
                            [[1],[1],[1],...],
                            [[2],[2],[2],...]]
            cat之后：
            targets.shape = [na,nt,41]
            targets = [[[image1,class1, x, y, w, h, x1, y1, v1,...,x17,y17,v17,0],
                        [image2,class2, x, y, w, h, x1, y1, v1,...,x17,y17,v17,0],
                        ...
                        [imageN,classN, x, y, w, h, x1, y1, v1,...,x17,y17,v17,0]],
                        [[image1,class1, x, y, w, h, x1, y1, v1,...,x17,y17,v17,1],
                        [image2,class2, x, y, w, h, x1, y1, v1,...,x17,y17,v17,1],
                        ...],
                        [[image1,class1, x, y, w, h, x1, y1, v1,...,x17,y17,v17,2],
                        [image2,class2, x, y, w, h, x1, y1, v1,...,x17,y17,v17,2],
                        ...]]
            这么做是为了纪录每个label对应的anchor。
        """
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        """
            定义每个grid偏移量，会根据标签在grid中的相对位置来进行偏移
        """
        g = 0.5  # bias
        """
            [0, 0]代表中间,
            [1, 0] * g = [0.5, 0]代表往左偏移半个grid， [0, 1]*0.5 = [0, 0.5]代表往上偏移半个grid，与后面代码的j,k对应
            [-1, 0] * g = [-0.5, 0]代代表往右偏移半个grid， [0, -1]*0.5 = [0, -0.5]代表往下偏移半个grid，与后面代码的l,m对应
            具体原理在代码后讲述
     
        """
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            # Detect层有四个检测头，每个检测头有三个候选框
            """
                原本yaml中加载的anchors.shape = [3,6],但在yolo.py的Detect中已经通过代码
                a = torch.tensor(anchors).float().view(self.nl, -1, 2)
                self.register_buffer('anchors', a) 
                将anchors进行了reshape。
                self.anchors.shape = [3,3,2]
                anchors.shape = [3,2]
            """
            anchors = self.anchors[i]
            # 现在gain [1,1,x,y,w,h,...]
            # p代表当前特征层的shape,默认是80*80，40*40，20*20
            if self.kpt_label:
                gain[2:40] = torch.tensor(p[i].shape)[19*[3, 2]]  # xyxy gain
            else:
                gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            """
                因为targets进行了归一化，默认在w = 1, h =1 的坐标系中，
                需要将其映射到当前输出层w = nx, h = ny的坐标系中。
            """
            t = targets * gain
            if nt:
                # Matches
                """
                    t[:, :, 4:6].shape = [na,nt,2] = [3,nt,2],存放的是标签的w和h
                    anchor[:,None] = [3,1,2]
                    r.shape = [3,nt,2],存放的是标签和当前层anchor的长宽比
                """
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                """
                    torch.max(r, 1. / r)求出最大的宽比和最大的长比，shape = [3,nt,2]
                    再max(2)求出同一标签中宽比和长比较大的一个，shape = [2，3,nt],之所以第一个维度变成2，
                    因为torch.max如果不是比较两个tensor的大小，而是比较1个tensor某一维度的大小，则会返回values和indices：
                        torch.return_types.max(
                            values=tensor([...]),
                            indices=tensor([...]))
                    所以还需要加上索引0获取values，
                    torch.max(r, 1. / r).max(2)[0].shape = [3,nt],
                    将其和hyp.yaml中的anchor_t超参比较，小于该值则认为标签属于当前输出层的anchor
                    j = [[bool,bool,....],[bool,bool,...],[bool,bool,...]]
                    j.shape = [3,nt]                 
                """
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                """
                    t.shape = [na,nt,41] 
                    j.shape = [3,nt]
                    假设j中有NTrue个True值，则
                    t[j].shape = [NTrue,7]
                    返回的是na*nt的标签中，所有属于当前层anchor的标签。             
                """
                t = t[j]  # 过滤掉比预设小的样本

                # Offsets
                """
                下面这段代码和注释可以配合代码后的图片进行理解。
                t.shape = [NTrue,41] 
                41:image,class,x, y, w, h, x1, y1, ,...,x17,y17,ai
                gxy.shape = [NTrue,2] 存放的是x,y,相当于坐标到坐标系左边框和上边框的记录
                gxi.shape = [NTrue,2] 存放的是w-x,h-y,相当于测量坐标到坐标系右边框和下边框的距离
            
                """
                gxy = t[:, 2:4]  # grid xy # target中心点的相对于左上角的坐标
                gxi = gain[[2, 3]] - gxy  # inverse  # target中心点相对于右下角的坐标

                '''
                因为grid单位为1，共nx*ny个gird
                gxy % 1相当于求得标签在第gxy.long()个grid中以grid左上角为原点的相对坐标，
                gxi % 1相当于求得标签在第gxy.long()个grid中以grid右下角为原点的相对坐标，
                下面这两行代码作用在于
                筛选中心坐标 左、上方偏移量小于0.5,并且中心点大于1的标签
                筛选中心坐标 右、下方偏移量小于0.5,并且中心点大于1的标签          
                j.shape = [NTrue], j = [bool,bool,...]
                k.shape = [NTrue], k = [bool,bool,...]
                l.shape = [NTrue], l = [bool,bool,...]
                m.shape = [NTrue], m = [bool,bool,...]
                '''
                # 判断是否把 左，上格子 也当作该目标进行训练
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # 判断是否把 右，下格子 也当作该目标进行训练
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                '''
                j.shape = [5,NTrue]
                t.repeat之后shape为[5,NTrue,7], 
                通过索引j后t.shape = [NOff,7],NOff表示NTrue + (j,k,l,m中True的总数量)
                torch.zeros_like(gxy)[None].shape = [1,NTrue,2]
                off[:, None].shape = [5,1,2]
                相加之和shape = [5,NTrue,2]
                通过索引j后offsets.shape = [NOff,2]
                这段代码的表示当标签在grid左侧半部分时，会将标签往左偏移0.5个grid，上下右同理。
                '''   

                # (torch.ones_like(j)中心网格的格子，永远为正，j，k,l,m 分别代表左上右下的格子
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # repeat函数 第一个参数是重复的次数， 第二个是列重复的倍数，第三个是行重复的倍数
                # 现在就是将t给重复五遍，分别得到中心，左，上，右，下
                t = t.repeat((5, 1, 1))[j]
                # offsets代表中心，左上右下的偏移量
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            '''
            t.shape = [NOff,7],(image,class,x,y,w,h,ai)
            '''
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            '''
            offsets.shape = [NOff,2]
            gxy - offsets为gxy偏移后的坐标，
            gxi通过long()得到偏移后坐标所在的grid坐标
            '''

            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            '''
            a:所有anchor的索引 shape = [NOff]
            b:标签所属image的索引 shape = [NOff]
            gj.clamp_(0, gain[3] - 1)将标签所在grid的y限定在0到ny-1之间
            gi.clamp_(0, gain[2] - 1)将标签所在grid的x限定在0到nx-1之间
            indices = [image, anchor, gridy, gridx] 最终shape = [nl,4,NOff]
            tbox存放的是标签在所在grid内的相对坐标，∈[0,1] 最终shape = [nl,NOff]
            anch存放的是anchors 最终shape = [nl,NOff,2]
            tcls存放的是标签的分类 最终shape = [nl,NOff]
            '''
        
            # a表示了是属于哪一个预测框
            a = t[:, -1].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            if self.kpt_label:
                for kpt in range(self.nkpt):
                    t[:, 6+2*kpt: 6+2*(kpt+1)][t[:,6+2*kpt: 6+2*(kpt+1)] !=0] -= gij[t[:,6+2*kpt: 6+2*(kpt+1)] !=0]
                tkpt.append(t[:, 6:-1])
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, tkpt, indices, anch
