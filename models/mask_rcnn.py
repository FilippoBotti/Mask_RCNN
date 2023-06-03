import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights
import torch.nn as nn
import torch

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from PIL import Image
from glob import glob

import os
import random
import numpy as np
import matplotlib.pyplot as plt
# def build_model(num_classes):
#     # load an instance segmentation model pre-trained on COCO
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

#     # get the number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#     # Stop here if you are fine-tunning Faster-RCNN

#     # now get the number of input features for the mask classifier
#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     hidden_layer = 256
#     # and replace the mask predictor with a new one
#     model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
#                                                        hidden_layer,
#                                                        num_classes)

#     return model

# model = build_model(NUM_CLASSES)
# print(model)
# opt.lr = 3456789


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

        # self.isTrain = opt.isTrain
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') 


        # if self.is_train: # only defined during training time
        #     self.criterion_loss = nn.CrossEntropyLoss()
        #     self.optimizer = torch.optim.SGD(self.model.params, lr=0.005,
        #                     momentum=0.9, weight_decay=0.0005)
        #     self.optimizers = [self.optimizer]



    # def set_input(self, data):
    #     images, targets = data
    #     self.images =  list(image.to(self.device) for image in images)
    #     self.targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        

    def forward(self, image, target=None):
        return self.model(image, target)

    # def backward(self):
    #     self.loss_dict = self.model(self.images, self.targets) # when given images and targets as input it will return the loss
    #     self.losses = sum(loss for loss in self.loss_dict.values())
    #     self.loss_value = self.losses.item()
    #     self.train_loss_list.append(self.loss_value)

    # def optimize_parameters(self):
    #     """Calculate losses, gradients, and update network weights; called in every training iteration"""
    #     # forward
    #     self.forward(self.in)      # compute fake images and reconstruction images.
    #     # G_A and G_B
    #     self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
    #     self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
    #     self.backward_G()             # calculate gradients for G_A and G_B
    #     self.optimizer_G.step()       # update G_A and G_B's weights
    #     # D_A and D_B
    #     self.set_requires_grad([self.netD_A, self.netD_B], True)
    #     self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
    #     self.backward_D_A()      # calculate gradients for D_A
    #     self.backward_D_B()      # calculate graidents for D_B
    #     self.optimizer_D.step()  # update D_A and D_B's weights


