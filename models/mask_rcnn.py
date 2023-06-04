import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights
import torch.nn as nn
import torch

import torch
import torchvision
import torch.nn as nn

class Mask_RCNN(nn.Module):
    def __init__(self, num_classes, args, hidden_layer=256):
        super(Mask_RCNN, self).__init__()
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

        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, num_classes)
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(self.in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

    def forward(self, image, target=None):
        return self.model(image, target)

   


