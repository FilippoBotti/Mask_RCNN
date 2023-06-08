import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights
import torch.nn as nn
import torch
from torchvision.models.detection import roi_heads
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Union
import torch
from torch import nn
import warnings
from torch.jit.annotations import Tuple, List, Dict, Optional

from models.roi_heads import CustomRoIHeads


'''
     (box_predictor): FastRCNNPredictor(
      (cls_score): Linear(in_features=1024, out_features=14, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=56, bias=True)
      )
'''
class FastRCNNPredictorWithAccessory(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.cls_accessory = nn.Linear(in_channels,1)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        accessory = self.cls_accessory(x)
        return scores, bbox_deltas, accessory



class Mask_RCNN(nn.Module):
    def __init__(self, num_classes, args, hidden_layer=256):
       
        super(Mask_RCNN, self).__init__()
        self.args = args
        if args.pretrained:
            if args.version == "V1":
                self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=True)
                print("Loaded pretrained weights V1", flush=True)
            if args.version == "V2":
                self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=True)
                print("Loaded pretrained weights V2", flush=True)
        else:
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn()
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels

        self.training = args.mode == "training"
        

        if args.cls_accessory:
            self.model.roi_heads.box_predictor = FastRCNNPredictorWithAccessory(self.in_features, num_classes)
        else: 
            self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, num_classes)
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(self.in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
        if self.args.cls_accessory:
            r = self.model.roi_heads
    # Initiate the custom roi_heads (see https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py#L492)
            new_roi_heads = CustomRoIHeads(r.box_roi_pool, r.box_head, r.box_predictor, 
                r.proposal_matcher.high_threshold, r.proposal_matcher.low_threshold, 
                r.fg_bg_sampler.batch_size_per_image, r.fg_bg_sampler.positive_fraction,
                r.box_coder.weights, r.score_thresh, r.nms_thresh, r.detections_per_img,
                r.mask_roi_pool, r.mask_head, r.mask_predictor,
                r.keypoint_roi_pool, r.keypoint_head, r.keypoint_predictor)

            self.model.roi_heads = new_roi_heads


    def forward(self, images, targets=None):
        return self.model(images,targets)
    
