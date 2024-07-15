import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, NMS, autoShape
from models.experimental import MixConv2d, CrossConv, C3
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr


try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


def remove_from_cuda(feat_map_list):
    detached_feature_maps = [[map.detach() for map in sublist] for sublist in feat_map_list[0]]
    del detached_feature_maps

def get_anns(ann,img):
        annotations_x1y1x2y2 = torch.zeros_like(ann, dtype=torch.int, device=img.device)
        annotations_x1y1x2y2[:, 0] = ann[:, 0]
        annotations_x1y1x2y2[:, 1] = ann[:, 1]
        annotations_x1y1x2y2[:, 2] = (ann[:, 2] - 0.5 * ann[:, 4]) * img.shape[2]  
        annotations_x1y1x2y2[:, 3] = (ann[:, 3] - 0.5 * ann[:, 5]) * img.shape[3]  
        annotations_x1y1x2y2[:, 4] = (ann[:, 2] + 0.5 * ann[:, 4]) * img.shape[2]  
        annotations_x1y1x2y2[:, 5] = (ann[:, 3] + 0.5 * ann[:, 5]) * img.shape[3]  
        return annotations_x1y1x2y2 
    
def plot(image, ann):
    bbox = get_anns(ann,image)
    img = image[:, :3, :, :]
    image = img.permute(0, 2, 3, 1).squeeze()
    fig, ax = plt.subplots(1)
    image = image.detach().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    ax.imshow(image)
    box = [bbox[0][2],bbox[0][3],bbox[0][4],bbox[0][5]]
    bbox = [tensor.item() for tensor in box]
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    output_image_path = 'datasets/output_image_feats.jpg'
    plt.savefig(output_image_path)


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self,targets, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.labels = targets

    def forward(self, x, labels):
        # sumi check x shape and you made changes here
        obj_feats = 0
        #object level feature extraction for contrastive loss, sumi you made changes here
        if labels is not None and self.training:
            # sumi check x shape and you made changes here
            
            #plot(obj_feats[0], labels)
            #plot(obj_feats[1], labels)
            #plot(obj_feats[2], labels)
            #norm_feats = []
            #for f in obj_feats:
            #    f_norm = F.layer_norm(f, f.shape[1:]) # normalize feats
            #    norm_feats.append(f_norm)
            #remove_from_cuda(obj_feats)
            #obj_feats = self.get_object_feats(norm_feats,labels) # object features and correspondiong labels
            
            
            #    mean = f.mean(dim=(2, 3), keepdim=True)
            #    std = f.std(dim=(2, 3), keepdim=True)
            #    batch_norm = nn.BatchNorm2d(num_features=f.shape[1])
            #    f_norm = batch_norm(f)
                
            #    norm_feats.append(f_norm)
            #remove_from_cuda(obj_feats)
            #obj_feats = self.get_object_feats(obj_feats,labels) # object features and correspondiong labels
            
            #obj_feats = self.get_object_feats_adaptive(obj_feats,labels)
            raw_feats = [t.clone() for t in x]
            feats = []
            for f in raw_feats:
                if f.shape[2] == 80:
                    upsampled_feats1 = F.interpolate(f, size=(100, 100), mode='bilinear', align_corners=False)
                    feats.append(upsampled_feats1)
                elif f.shape[2] == 40:
                    upsampled_feats2 = F.interpolate(f, size=(100, 100), mode='bicubic', align_corners=False)
                    feats.append(upsampled_feats2)
                elif f.shape[2] == 20:
                    upsampled_feats3 = F.interpolate(f, size=(100, 100), mode='bicubic', align_corners=False)
                    feats.append(upsampled_feats3)
            obj_feats = self.get_obj_feats_by_index(feats, labels)
            #remove_from_cuda(raw_feats)
            #remove_from_cuda(feats)
            
            
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        pred  = x if self.training else (torch.cat(z, 1), x)
        return pred , obj_feats
        #return obj_feats, x  if self.training else (torch.cat(z, 1), x) #pred , obj_feats # sumi your changes

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    
    
    """ 
        extract object level features from the feature map
        by simple indexing. Then apply global average pool
        layer.
    """
    
    def get_obj_feats_by_index(self, feat, labels):
        obj_feat_list = [] #list containing 3 sublists at three levels, P2,P3,P4
        ground_truth = []
        for i in range(len(feat)):
            obj_feat, gt = self.get_feats(feat[i],labels)
            obj_feat_list.append(obj_feat) # list containig featuires at all three levels
            ground_truth.append(gt)
        return (obj_feat_list, ground_truth) # obj_fetures is a list of 3 sublists containing total number of objects in the total batch with its corresponding class
    
    def get_feats(self,f_map,gt):
        roi_feats = []
        rois_scaled = self.get_anns(gt, f_map)
        classes = gt[:,1].detach().cpu()
        for i in range(f_map.shape[0]): #batch size
            feat = f_map[i:i+1]
            rois = rois_scaled[rois_scaled[:, 0] == i].clone() # take gt corresponding to image
            if rois.numel() != 0:# if there is no object in this batch element
                #rois_noclass =  rois[:, [0, 2, 3, 4, 5]]
                for bbox_non_adjust in rois: #[batch, x1, y1, x2, y2]
                    bbox = self.adjust_float_bbox(bbox_non_adjust, feat)
                    obj_feat = feat[:,:,bbox[3]:bbox[5], bbox[2]:bbox[4]]
                    obj_pooled = F.adaptive_avg_pool2d(obj_feat, (1, 1))
                    obj_pooled = obj_pooled.view(obj_pooled.size(0),obj_pooled.size(1))
                    if torch.any(torch.isnan(obj_pooled)):
                        print('nan')
                    if torch.any(torch.isinf(obj_pooled)):
                        print('inf')
                    roi_feats.append(obj_pooled)
                    
        return roi_feats, classes
    
    def adjust_float_bbox(self,bbox,img):
        bbox[2] = torch.round(bbox[2])
        bbox[3] = torch.round(bbox[3])
        bbox[4] = torch.round(bbox[4])
        bbox[5] = torch.round(bbox[5])
        bbox = self.adjust_bbox(bbox.to(torch.int),img)
        return bbox
        
    
    def adjust_bbox(self,bbox, img):
        if bbox[2] == -1:
            bbox[2] = 0
        if bbox[3] == -1:
            bbox[3] = 0
            
        if bbox[4] - bbox[2] == 0: #width = 0
            if bbox[4] < img.shape[3]:
                bbox[4] = bbox[4]+1
            else:
                bbox[2] = bbox[2]-1
        if bbox[5] - bbox[3] == 0: #height = 0
            if bbox[5] < img.shape[2]: 
                bbox[5] = bbox[5]+1
            else:
                bbox[3] = bbox[3]-1
        return bbox
    
    
    def get_anns(self,ann,img):# ann:(batch_index,class, x_center, y_center, width, height)
        annotations_x1y1x2y2 = torch.zeros_like(ann, dtype=torch.float32, device=img.device)
        annotations_x1y1x2y2[:, 0] = ann[:, 0]
        annotations_x1y1x2y2[:, 1] = ann[:, 1]
        annotations_x1y1x2y2[:, 2] = (ann[:, 2] - 0.5 * ann[:, 4]) * img.shape[2]  # x1 = x_center - 0.5 * width
        annotations_x1y1x2y2[:, 3] = (ann[:, 3] - 0.5 * ann[:, 5]) * img.shape[3]  # y1 = y_center - 0.5 * height
        annotations_x1y1x2y2[:, 4] = (ann[:, 2] + 0.5 * ann[:, 4]) * img.shape[2]  # x2 = x_center + 0.5 * width
        annotations_x1y1x2y2[:, 5] = (ann[:, 3] + 0.5 * ann[:, 5]) * img.shape[3]  # y2 = y_center + 0.5 * height
        return annotations_x1y1x2y2 # ann:(batch_index,class, x1, y1, x2, y2)
    
    
    
    #helper to extract object features
    def get_object_feats(self, feat, labels):
        obj_feat_list = [] #list containing 3 sublists at three levels, P2,P3,P4
        ground_truth = []
        #roi_align = ROIAlign(output_size=(7,7))
        for i in range(len(feat)):
            obj_feat, gt = self.roi(feat[i], labels)
            obj_feat_list.append(obj_feat) # list containig featuires at all three levels
            ground_truth.append(gt)
        return (obj_feat_list, ground_truth) # obj_fetures is a list of 3 sublists containing total number of objects in the total batch with its corresponding class
    
    def roi(self,feature_map, rois):
        roi_feats = []
        #rois_scaled = rois.clone()
        #rois_scaled = [torch.Tensor([[i, x, y, x + w, y + h] for x, y, w, h in img_gt]) for i, img_gt in enumerate(rois_scaled)]
        #rois_scaled[:, 2:] *= torch.tensor([feature_map.shape[3], feature_map.shape[2], feature_map.shape[3], feature_map.shape[2]])
        #rois_scaled = self.get_annotations(rois_scaled, feature_map)
        rois_scaled = self.get_anns(rois, feature_map)
        gt = rois[:,1].detach().cpu()
        for i in range(feature_map.shape[0]): #batch size
            rois = rois_scaled[rois_scaled[:, 0] == i] # take gt corresponding to image
            if rois.numel() == 0:# if there is no object in this batch element
                continue
            rois_noclass =  rois[:, [0, 2, 3, 4, 5]]
            # Apply ROI align
            roi_aligned_features = roi_align(feature_map[i:i+1], rois_noclass, output_size=(5,5))
            
            if torch.any(torch.isnan(roi_aligned_features)):
                #roi_aligned_features = roi_align(feature_map[i:i+1], rois_noclass, output_size=self.output_size)
                #roi_aligned_features = roi_align(feature_map[i:i+1], rois_noclass, output_size=self.output_size)
                print('nan')
            if torch.any(torch.isinf(roi_aligned_features)):
                print('inf')
            global_avg_pooled = F.adaptive_avg_pool2d(roi_aligned_features, (1, 1))
            global_avg_pooled = global_avg_pooled.view(global_avg_pooled.size(0),global_avg_pooled.size(1))
            aligned_features = torch.split(global_avg_pooled, 1)
            for f in aligned_features:
                if f.numel() == 0:
                    print("empty tensor")
            #roi_aligned_features = torch.split(roi_aligned_features, 1) uncomment the lines if noglobal average pooled features are required
            roi_feats.extend(aligned_features)
            #roi_aligned_features = torch.stack([roi_align(fm.unsqueeze(0), roi, output_size=self.output_size) for fm, roi in zip(feature_map, rois)])
        #aligned_features = nn.functional.grid_sample(feature_map, 
        #                                             nn.functional.affine_grid(
        #                                                 self.get_affine_matrix(rois_scaled, feature_map.shape[0]),feature_map.size()))
        return roi_feats, gt
        
        
    #helper to extract object features
    def get_object_feats_adaptive(self, feat, labels):
        obj_feat_list = [] #list containing 3 sublists at three levels, P2,P3,P4
        ground_truth = []
        for i in range(len(feat)):
            feat_size = feat[0][0][0][0].shape
            out_size = int(feat_size[0] / len(labels))
            if i==0:
                out_size = min(out_size, 11)
            elif i==1:
                out_size = min(out_size, 7)
            elif i==2:
                out_size = min(out_size, 3)
            roi_align= ROIAlign(output_size=(out_size,out_size))
            obj_feat, gt = roi_align(feat[i], labels)
            obj_feat_list.append(obj_feat) # list containig featuires at all three levels
            ground_truth.append(gt)
        return (obj_feat_list, ground_truth) # obj_fetures is a list of 3 sublists containing total number of objects in the total batch with its corresponding class
    
        
    
    

#helper to extract object features outside the yolov5 model
def get_object_feats_outside_model(feat, labels):
    obj_feat_list = [] #list containing 3 sublists at three levels, P2,P3,P4
    ground_truth = []
    roi_align = ROIAlign(output_size=(7,7))
    for i in range(len(feat)):
        obj_feat, gt = roi_align(feat[i], labels)
        obj_feat_list.append(obj_feat) # list containig featuires at all three levels
        ground_truth.append(gt)
    return (obj_feat_list, ground_truth) # obj_fetures is a list of 3 sublists containing total number of objects in the total batch with its corresponding class


class ClassifyObjFeats(nn.Module):
    def __init__(self, input_size, nc):
        super(ClassifyObjFeats, self).__init__()
        hidden_size = 50
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()  # Replace with LeakyReLU
        self.fc2 = nn.Linear(hidden_size, nc)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out

# Define a simple ROI Align layer, sumi your changes here
class ROIAlign(nn.Module):
    def __init__(self, output_size):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        #self.roialign =  RoIAlign(self.output_size)
    def forward(self, feature_map, rois):
        # ROIs are in the format (batch_index,class, x_center, y_center, width, height)
        roi_feats = []
        #rois_scaled = rois.clone()
        #rois_scaled = [torch.Tensor([[i, x, y, x + w, y + h] for x, y, w, h in img_gt]) for i, img_gt in enumerate(rois_scaled)]
        #rois_scaled[:, 2:] *= torch.tensor([feature_map.shape[3], feature_map.shape[2], feature_map.shape[3], feature_map.shape[2]])
        #rois_scaled = self.get_annotations(rois_scaled, feature_map)
        rois_scaled = self.get_anns(rois, feature_map)
        gt = rois[:,1].detach().cpu()
        for i in range(feature_map.shape[0]): #batch size
            rois = rois_scaled[rois_scaled[:, 0] == i] # take gt corresponding to image
            rois_noclass =  rois[:, [0, 2, 3, 4, 5]]
            # Apply ROI align
            roi_aligned_features = roi_align(feature_map[i:i+1], rois_noclass, output_size=self.output_size)
            if torch.any(torch.isnan(roi_aligned_features)):
                #roi_aligned_features = roi_align(feature_map[i:i+1], rois_noclass, output_size=self.output_size)
                #roi_aligned_features = roi_align(feature_map[i:i+1], rois_noclass, output_size=self.output_size)
                print('nan')
            if torch.any(torch.isinf(roi_aligned_features)):
                print('inf')
            global_avg_pooled = F.adaptive_avg_pool2d(roi_aligned_features, (1, 1))
            global_avg_pooled = global_avg_pooled.view(global_avg_pooled.size(0),global_avg_pooled.size(1))
            aligned_features = torch.split(global_avg_pooled, 1)
            #roi_aligned_features = torch.split(roi_aligned_features, 1) uncomment the lines if noglobal average pooled features are required
            roi_feats.extend(aligned_features)
            #roi_aligned_features = torch.stack([roi_align(fm.unsqueeze(0), roi, output_size=self.output_size) for fm, roi in zip(feature_map, rois)])
        #aligned_features = nn.functional.grid_sample(feature_map, 
        #                                             nn.functional.affine_grid(
        #                                                 self.get_affine_matrix(rois_scaled, feature_map.shape[0]),feature_map.size()))
        return roi_feats, gt #torch.stack(self.roi_feats, dim=0) 

    def get_affine_matrix(self, rois, batch):
        # Helper function to calculate the affine matrix for grid sampling
        batch_size = batch
        theta = torch.zeros((batch_size, 2, 3), dtype=torch.float32, device=rois.device)
        theta[:, 0, 0] = rois[:, 4] / self.output_size[1]
        theta[:, 1, 1] = rois[:, 5] / self.output_size[0]
        theta[:, :, 2] = torch.stack([rois[:, 2] - 0.5 * rois[:, 4], rois[:, 3] - 0.5 * rois[:, 5]], dim=1)
        return theta
    
    
    def get_anns(self,ann,img):# ann:(batch_index,class, x_center, y_center, width, height)
        annotations_x1y1x2y2 = torch.zeros_like(ann, dtype=torch.float16, device=img.device)
        annotations_x1y1x2y2[:, 0] = ann[:, 0]
        annotations_x1y1x2y2[:, 1] = ann[:, 1]
        annotations_x1y1x2y2[:, 2] = (ann[:, 2] - 0.5 * ann[:, 4]) * img.shape[2]  # x1 = x_center - 0.5 * width
        annotations_x1y1x2y2[:, 3] = (ann[:, 3] - 0.5 * ann[:, 5]) * img.shape[3]  # y1 = y_center - 0.5 * height
        annotations_x1y1x2y2[:, 4] = (ann[:, 2] + 0.5 * ann[:, 4]) * img.shape[2]  # x2 = x_center + 0.5 * width
        annotations_x1y1x2y2[:, 5] = (ann[:, 3] + 0.5 * ann[:, 5]) * img.shape[3]  # y2 = y_center + 0.5 * height
        return annotations_x1y1x2y2



class Model(nn.Module):
    def __init__(self,targets, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        
        #define classifier
        self.classifier_l0 = ClassifyObjFeats(320, nc)
        self.classifier_l1 = ClassifyObjFeats(640, nc)
        self.classifier_l2 = ClassifyObjFeats(1280, nc)
        # Define model
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward( torch.zeros(1, ch, s, s), targets)[0]])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, targets, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi, targets)[0][0]  # forward sumi you chnages 0 to 1
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # obj_feats, augmented inference, train,  , sumi you chnages this
        else:
            return self.forward_once(x, targets, profile)  # single-scale inference, train

    def forward_once(self, x, targets, profile=False): # x input 
        y, dt = [], []  # outputs
        #cur_layer = 0   # mark cur_layer num if 9 detach sumi you commented it out 
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
                
            if isinstance(m, Detect): #sumi you added this if else block
                x,obj_feats  = m(x,targets) 
            else:
                x = m(x)  # run 
            #if cur_layer == 9:
                #self.backbone_feature = x.detach() # output backbone feature  output [B, 1280, 14, 21]
                #print(self.backbone_feature.shape)
            #cur_layer += 1 sumi you commented theses lines
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x, obj_feats   #sumi you added this, None just to remove the error

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(4 / (640 / s) ** 2)  # obj (8 objects per 640 image) sumi you made changes here, 8-->4
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[2], int):  # number of anchors sumi you chnages 1 to 2
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
