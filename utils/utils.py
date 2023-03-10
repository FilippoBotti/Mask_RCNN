from tqdm import tqdm
import torch
from PIL import Image
import utils.transforms as T
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.utils import draw_segmentation_masks, make_grid
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'

MASK_COLORS=[
(255, 0, 0) ,# Red
(0, 255, 0) ,# Green
(0, 0, 255) ,# Blue
(255, 255, 0) ,# Yellow
(255, 0, 255) ,# Magenta
(0, 255, 255) ,# Cyan
(128, 0, 0) ,# Maroon
(0, 128, 0) ,# Olive
(0, 0, 128) ,# Navy
(128, 128, 0) ,# Olive Drab
(128, 0, 128) ,# Purple
(0, 128, 128) ,# Teal
(192, 192, 192) ,# Silver
]



def show(imgs):
    if(len(imgs)!=6):
        print("Not correct size", len(imgs))
        return
    
    print('RED   bag\nGREEN	belt\nBLUE	boots\nYELLOW	footwear\nMAGENTA	outer\nCIANO	dress\nMARRONE	sunglasses\nOLIVA	pants\nNAVY top\nOLIVE DRAB	shorts\nVIOLA	skirt\nTEAL	headwear\nARGENTO	scarf\n')
    plt.subplot(2,3,1)
    plt.imshow(np.squeeze(imgs[0].cpu().numpy()).transpose(1,2,0))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(2,3,2)
    plt.imshow(imgs[1].cpu().numpy().transpose(1,2,0))
    plt.title('Prediction')
    plt.axis('off')
    plt.subplot(2,3,3)
    plt.imshow(imgs[2].cpu().numpy().transpose(1,2,0))
    plt.title('Original Mask')
    plt.axis('off')

    plt.subplot(2,3,4)
    plt.imshow(imgs[3].cpu().numpy().transpose(1,2,0))
    plt.axis('off')
    plt.subplot(2,3,5)
    plt.imshow(imgs[4].cpu().numpy().transpose(1,2,0))
    plt.axis('off')
    plt.subplot(2,3,6)
    plt.imshow(imgs[5].cpu().numpy().transpose(1,2,0))
    plt.axis('off')
    plt.show()

def matplotlib_imshow(img, one_channel=False):
    img = img.cpu()
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    
    to_pil = transforms.ToPILImage()
    
    if one_channel:
        npimg = img.numpy()
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(to_pil(img))

def visualize_sample(image, target, classes):
    original_img = image.byte()
    mask=original_img.clone()
    img = np.array(to_pil(image.byte())).copy()
    for box_num in range(len(target['boxes'])):
        box = target['boxes'][box_num]
        label = classes[target['labels'][box_num]]
        

        cv2.rectangle(
            img, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 2
        )
        cv2.putText(
            img, label, (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
    mask = draw_segmentation_masks(original_img,target['masks']>0)
    images = [original_img, torch.from_numpy(img).permute(2,0,1), mask]

    return images

def visualize_bbox(image, prediction, target, classes):
    image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).byte()
    bbox_img = np.array(to_pil(torch.zeros_like(image))).copy()
    real_bbox_img = np.array(to_pil(torch.zeros_like(image))).copy()
    for i in range(len(prediction[0]['boxes'])):
        box=prediction[0]['boxes'][i].detach().cpu()
        scr=prediction[0]['scores'][i].detach().cpu()
        label = classes[prediction[0]['labels'][i].detach().cpu()]
        if scr>0.8:
            #print(box)
            cv2.rectangle(
            bbox_img, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 2
        )

            cv2.putText(
            bbox_img, label, (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
    for box_num in range(len(target['boxes'])):
        box = target['boxes'][box_num]
        label = classes[target['labels'][box_num]]

        cv2.rectangle(
            real_bbox_img, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 2
        )
        cv2.putText(
            real_bbox_img, label, (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
    images = [image, torch.from_numpy(bbox_img).permute(2,0,1), torch.from_numpy(real_bbox_img).permute(2,0,1)]
    return images

def visualize_mask(image, prediction, target):
    image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).byte()
    mask = torch.zeros_like(image)
    real_mask = torch.zeros_like(image)


    for i in range(len(prediction[0]['masks'])):
        msk=prediction[0]['masks'][i,0].detach().cpu()
        scr=prediction[0]['scores'][i].detach().cpu()
        label = prediction[0]['labels'][i].detach().cpu()
        if scr>0.8:
            mask = draw_segmentation_masks(mask,msk>0.5,1, MASK_COLORS[label-1])
    
    for i in range(len(target['masks'])):
        real_mask = draw_segmentation_masks(real_mask,target['masks'][i]>0,1, MASK_COLORS[target['labels'][i]-1])
    results = [image, mask, real_mask]
    return results

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return img

to_tensor = T.PILToTensor()
to_pil = transforms.ToPILImage()

def collate_fn(batch):
    return tuple(zip(*batch)) 


def get_train_transform():
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.RandomHorizontalFlip(p=0.5),
    ])
# define the validation transforms
def get_valid_transform():
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])

def save_model(epoch, model, optimizer):
    """
    https://drive.google.com/file/d/1W_GRxnTfgGarIvobS7CJDCjcO9Hc6drg/view
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'res/mask_rcnn.pth')
