import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
from tqdm import tqdm
import sys
from models.mask_rcnn import Mask_RCNN
from utils.utils import  visualize_mask, show
from torchvision.utils import draw_segmentation_masks, make_grid
import matplotlib.pyplot as plt
import torchvision
import cv2, random, numpy as np

class Solver(object):
    """Solver for training and testing."""

    def __init__(self, train_loader, valid_loader, device, writer, classes, args):
        """Initialize configurations."""
        self.args = args
        self.model_name = 'modanet_maskRCNN_{}.pth'.format(self.args.model_name)

        # Define the model
        self.classes = classes
        self.num_classes = len(self.classes)

        self.net = Mask_RCNN(self.num_classes).to(device)

        # load a pretrained model
        if self.args.resume_train == True:
            self.load_model()
        
        # Choose optimizer
        if self.args.opt == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr)
        elif self.args.opt == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)

        self.epochs = self.args.epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = device
        self.writer = writer

    def save_model(self):
        # if you want to save the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        torch.save(self.net.state_dict(), check_path)
        print("Model saved!")

    def load_model(self):
        # function to load the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path))
        print("Model loaded!")
    
    def train(self):
        # self.net.train()
        # for epoch in range(self.epochs):
        #     print(f"\nEPOCH {epoch+1} of {self.epochs}")
        #     running_loss = 0.0
        #     # start timer and carry out training and validation
        #     start = time.time()
        #     print('Solver Training')
        #     train_loss_list = []
            
        #     # initialize tqdm progress bar
        #     prog_bar = tqdm(self.train_loader, total=len(self.train_loader))
        #     loss_dict_tb = {
        #         "loss_classifier": 0,
        #         "loss_box_reg": 0,
        #         "loss_mask": 0,
        #         "loss_objectness": 0,
        #         "loss_rpn_box_reg": 0
        #     }

        #     for i, data in enumerate(prog_bar):
        #         self.optimizer.zero_grad()
        #         images, targets = data
        #         images =  list(image.to(self.device) for image in images)
        #         targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        #         loss_dict = self.net(images, targets) # when given images and targets as input it will return the loss
        #         losses = sum(loss for loss in loss_dict.values())
        #         loss_value = losses.item()
        #         train_loss_list.append(loss_value)
        #         losses.backward()
        #         self.optimizer.step()
                
        #         prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        #         running_loss += loss_value
               
        #         for loss in loss_dict:
        #             loss_dict_tb[loss] += loss_dict[loss].item()

        #         del losses, loss_dict, loss_value

        #         if i % self.args.print_every == self.args.print_every - 1:  
        #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / self.args.print_every:.3f}')

        #             self.writer.add_scalar('training loss',
        #                 running_loss / self.args.print_every,
        #                 epoch * len(self.train_loader) + i)
                    
        #             for loss in loss_dict_tb:
        #                 self.writer.add_scalar(loss,
        #                 loss_dict_tb[loss]/ self.args.print_every,
        #                 epoch * len(self.train_loader) + i)

        #             running_loss = 0.0
        #             loss_dict_tb = {
        #                 "loss_classifier": 0,
        #                 "loss_box_reg": 0,
        #                 "loss_mask": 0,
        #                 "loss_objectness": 0,
        #                 "loss_rpn_box_reg": 0
        #             }

        #     val_loss_list = self.validate()

        #     print(f"Epoch #{epoch+1} train loss: {sum(train_loss_list)//len(self.train_loader):.3f}")   
        #     print(f"Epoch #{epoch+1} validation loss: {sum(val_loss_list)//len(self.valid_loader):.3f}")  

        #     self.writer.add_scalar('validation loss',
        #                 sum(val_loss_list)//len(self.valid_loader),epoch)
        #     end = time.time()
        #     print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        #     self.save_model()
        # self.writer.flush()
        # self.writer.close()
        # print('Finished Training')  
        self.test() 
    
    def validate(self):
        print('Validating')
        val_itr = 0
        val_loss_list = []
        # initialize tqdm progress bar
        prog_bar = tqdm(self.valid_loader, total=len(self.valid_loader))
        loss_value = 0
        for i, data in enumerate(prog_bar):
            images, targets = data
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                loss_dict = self.net(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            val_loss_list.append(loss_value)
            val_itr += 1
            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}\n\n")
        self.net.train()
        return val_loss_list
    
    def test(self):
        print("Testing")
        for data in self.valid_loader:
            images, targets = data
            self.net.eval()
            prediction = self.net([images[0]])
            # print(predictions)
            results = visualize_mask(images[0],prediction,targets[0])
            show(results)