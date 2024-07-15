# Loss functions


import torch
import torch.nn as nn

from models.yolo import ClassifyObjFeats
from models.yolo import Model

import numpy as np
from utils.general import bbox_iou
from utils.torch_utils import is_parallel
from itertools import combinations


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
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
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
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
        

# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self,nc, margin=1.3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.nc = nc
    def __call__(self, model, obj_feat, bs, dvc):
        loss = 0
        temp = 0.01
        obj_level_feat, obj_gt = obj_feat
        #dvc = obj_level_feat[0][0].device
        obj_groups = self.get_obj_groups(obj_gt) # dictionary with keys as classes and values as list of indices
        
        #classification loss
        level_class_loss = self.classify(model, obj_level_feat, obj_groups)
        
        level_sim_loss = self.compute_similarity(obj_level_feat, obj_groups) 
        level_dis_loss = self.compute_dissimilarity(obj_level_feat, obj_groups)
        #sim_loss = sum(level_sim_loss.values()) if torch.is_tensor(sum(level_sim_loss.values())) else torch.tensor(sum(level_sim_loss.values()), requires_grad=True, device=dvc).view(1)
        #dis_loss = sum(level_dis_loss.values()) if torch.is_tensor(sum(level_dis_loss.values())) else torch.tensor(sum(level_dis_loss.values()), requires_grad=True, device=dvc).view(1) 
        sim_loss =  self.average(level_sim_loss,temp,dvc)
        dis_loss =  self.average(level_dis_loss,temp,dvc)
        clas_loss = self.average(level_class_loss,temp,dvc)
        #torch.tensor(0.0, device=dvc).view(1)# 
        loss = 0.002*sim_loss + 0.005*dis_loss + 0.005*clas_loss
        return loss, torch.cat((sim_loss, dis_loss,clas_loss,loss)).detach() #clas_loss,
    
        #return loss*bs , torch.cat((sim_loss.view(1), dis_loss.view(1),loss.view(1))).detach()
    def average(self,dict,temp, dvc):
        for k,v in dict.items():
            if torch.is_tensor(v) and torch.isnan(v):
                dict[k] = torch.tensor(temp, requires_grad=True, device=dvc).view(1)
            elif not torch.is_tensor(v):
                dict[k] = torch.tensor(v, requires_grad=True, device=dvc).view(1)
        #return dict['0'] 
        #return sum(dict.values())  
        l = list(dict.values())
        s = torch.stack(l)
        return torch.mean(s, dim=0) 
       
            
        
        
                
    def has_nan(self,tensor):
        if torch.isnan(tensor):
            nan_m = torch.isnan(tensor)
            tensor = tensor.clone()  # Create a copy to avoid modifying the original tensor
            tensor[nan_m] = 0.01
        return tensor
    
    def compute_dissimilarity(self,obj_feat,obj_groups):
        d_loss = {'0':0.0, '1':0.0, '2':0.0}
        l = 0
        for f in obj_feat: #mean features at three diffrent levels
            loss = 0.0
            count = 0.0
            m_feats = {}
            if len(obj_groups.keys())>1:
                for k,v in obj_groups.items(): #constrauct mean feats dictionary
                    if len(v) > 1:
                        m_feats = self.get_mean_feats(f,v,k,m_feats)
                    elif len(v)==1:
                        if k not in m_feats.keys():
                            m_feats[k] = f[v[0]]
                if len(m_feats.keys()) == 2:
                    euclidean_distance = torch.nn.functional.cosine_similarity(m_feats[list(m_feats.keys())[0]], m_feats[list(m_feats.keys())[1]])#torch.nn.functional.pairwise_distance(m_feats[list(m_feats.keys())[0]], m_feats[list(m_feats.keys())[1]], keepdim=False)
                    #print('dissimilarity when 2 objects: ', euclidean_distance)
                    loss_contrastive = torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
                    loss=loss_contrastive
                elif len(m_feats.keys()) > 2:
                    comb = combinations([i for i in m_feats.keys()],2) #genrate combinations
                    for c in comb:
                        euclidean_distance = torch.nn.functional.cosine_similarity(m_feats[c[0]], m_feats[c[1]]) #torch.nn.functional.pairwise_distance(m_feats[c[0]], m_feats[c[1]], keepdim=True)
                        #print('dissimilarity when >2 objects: ', euclidean_distance)
                        loss_contrastive = torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
                        loss+=loss_contrastive
                        count+=1
                d_loss[str(l)] = (loss/count).view(1) if count > 0.0 else loss
            l+=1
        del l
        del loss
        return d_loss       
        
    
    
    def compute_similarity(self,obj_feats, obj_groups):
        c_loss = {'0':0.0,'1':0.0,'2':0.0} #loss at three feature levels
        l = 0 #feature level
        for f in obj_feats:
            loss = 0.0
            for k, v in obj_groups.items():
                count = 0.0
                local_loss = 0.0
                if len(v) == 2:
                    euclidean_distance = torch.nn.functional.cosine_similarity(f[v[0]], f[v[1]])#torch.nn.functional.pairwise_distance(f[v[0]], f[v[1]], keepdim=True)
                    #print('similarity when 2 objects: ', euclidean_distance)
                    loss_contrastive = torch.pow(euclidean_distance, 2)
                    local_loss+=loss_contrastive
                    count+=1.0
                elif len(v) > 2:
                    comb = combinations([i for i in v],2) #genrate combinations
                    for c in comb:
                        euclidean_distance = torch.nn.functional.cosine_similarity(f[c[0]], f[c[1]])#torch.nn.functional.pairwise_distance(f[c[0]], f[c[1]], keepdim=True)
                        #print('similarity when >2 objects: ', euclidean_distance)
                        loss_contrastive = torch.pow(euclidean_distance, 2)
                        local_loss+=loss_contrastive
                        count+=1.0
                local = (local_loss/count).view(1) if count > 0.0 else local_loss
                loss +=  local
            c_loss[str(l)] = loss
            l+=1
        del l
        del loss
        return c_loss
              
    
    
    """
        keys are the object classes and 
        values,a list of three(at three feature maps)
        mean feat
    """
    def get_mean_feats(self,feat,indices,clas,mfeat):
        mean = torch.mean(torch.stack([feat[i] for i in indices], dim=0),dim=0, dtype=torch.float64)
        if clas not in mfeat.keys():
            mfeat[clas] = mean
        else:
            "Error: class already present"
        return mfeat
            
            
    """
        keys are the object classes and 
        values,a list of three(at three feature maps)
        mean feat
    """
    def get_mean_feat_at_levels(self,obj_groups,features):
        mfeat = {} 
        for f in features:
            for k in obj_groups.keys():
                indices = obj_groups[k]
                mean = torch.mean(torch.stack([f[i] for i in indices], dim=0), 
                                                   dim=0, dtype=torch.float64)
                if k in mfeat.keys():
                    mfeat[k].append(mean)
                else:
                    mfeat[k] = [mean]
        return mfeat
    
    
    def get_obj_groups(self,obj_gt): 
        group_classes = {}
        gt = obj_gt[0].numpy()
        for i in range(len(gt)):
            if gt[i] in group_classes.keys():
                group_classes[gt[i]].append(i)
            else:
                group_classes[gt[i]] = [i]
        return group_classes
    
    
    def classify(self, model, obj_feats, obj_groups):
        class_loss = {'0':0.0,'1':0.0,'2':0.0} #loss at three feature levels
        criterion = nn.CrossEntropyLoss()
        l = 0 #feature level
        for f in obj_feats:
            loss = 0.0
            #input_size = f[0].shape[-1]
            if l == 0:
                clasifier = model.classifier_l0
            elif l == 1:
                clasifier = model.classifier_l1
            elif l == 2:
                clasifier = model.classifier_l2
            for k, v in obj_groups.items():
                count = 0.0
                local_loss = 0.0
                for i in v:
                    pred = clasifier(f[i])
                    gt = torch.eye(self.nc)[int(k)]
                    clasloss = criterion(pred, gt.unsqueeze(0).cuda() )
                    local_loss += clasloss
                    count+=1.0
                local = (local_loss/count).view(1)
                loss +=  local
            class_loss[str(l)] = loss
            l+=1
        del l
        del loss
        return class_loss
    
    
    def compute_dissimilarity_old(self,mfeat,label=1, comb_idx=None):
        d_loss = {'0':0, '1':0, '2':0}
        for i in range(3): #mean features at three diffrent levels
            if comb_idx is not None:
                comb_loss = 0
                for c in comb_idx:
                    euclidean_distance = torch.nn.functional.pairwise_distance(mfeat[c[0]], mfeat[c[1]], keepdim=True) # torch.nn.functional.cosine_similarity(mfeat[c[0]], mfeat[c[1]], size_average=True)
                    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                     (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
                    
                d_loss[i] = loss_contrastive 
            else:
                euclidean_distance = torch.nn.functional.pairwise_distance(mfeat[[mfeat.keys()][0]], mfeat[[mfeat.keys()][1]], keepdim=True) # torch.nn.functional.cosine_similarity(mfeat[[mfeat.keys()][0]], mfeat[[mfeat.keys()][1]], size_average=True)
                loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                     (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
                d_loss[i] = loss_contrastive
        return d_loss
        
   
    
        """
        gt is a super list of 3 sublists containing 
        groundtruth values for the feature maps at three levels
        hence the ground truth will be same, so we will take
        the first sublist of ground truths
        """
    def compute_similarity_old(self,feat,label=0,obj_idx=None, comb_idx=None): # 0 label for similar class
        c_loss = {'0':0,'1':0,'2':0} #loss at three feature levels
        if comb_idx is not None:
            for c in comb_idx:
                euclidean_distance = torch.nn.functional.pairwise_distance(feat[c[0]], feat[c[1]], keepdim=True)
                loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                     (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
                c_loss+=loss_contrastive
            return c_loss
        elif obj_idx is not None:
            for i in range(3):
                euclidean_distance = torch.nn.functional.pairwise_distance(feat[i][obj_idx[0]], feat[i][obj_idx[1]], keepdim=True)
                loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                     (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
                c_loss[i] = loss_contrastive
            return c_loss
  
    


def compute_loss(p, targets, model):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters
   

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)
   
    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no == 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch
