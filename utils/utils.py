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

CATEGORY_LABELS = [
    "bag",
    "belt",
    "boots",
    "footwear",
    "outer",
    "dress",
    "sunglasses",
    "pants",
    "top",
    "shorts",
    "skirt",
    "headwear",
    "scarf & tie"
]

ACCESSORIES_ID_LIST = [1,2,7,12,13]

def show(imgs):
    # for i in imgs:    

    # concatenation = np.concatenate((imgs[0],imgs[1],imgs[2]), axis=1)
    # plt.imshow(concatenation)  
    # plt.show()
    #writer.add_image('four_fashion_mnist_images', concatenation)
# show images
    # matplotlib_imshow(img_grid, one_channel=True)
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

def predicted_mask(image, prediction):
    image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).byte()
    mask = torch.zeros_like(image)

    for index, element in enumerate(prediction[0]['boxes']):
        msk=prediction[0]['masks'][index,0].detach().cpu()
        scr=prediction[0]['scores'][index].detach().cpu()
        label = prediction[0]['labels'][index].detach().cpu()
        if scr>0.8:
            mask = draw_segmentation_masks(mask,msk>0.5,1, MASK_COLORS[label-1])
    
    results = [image, mask]
    return [mask.permute(1,2,0).numpy()]

def predicted_accessories_and_labels(image, prediction, target, cls_accessory):
    predicted_results = []
    targets = []
    for index, element in enumerate(prediction[0]['boxes']):
        scr= prediction[0]['scores'][index].detach().cpu()
        result = {}
        if scr>0.8:
            label = prediction[0]['labels'][index].detach().cpu()
            result['predicted_label'] = CATEGORY_LABELS[label-1]
            result['label_score'] = scr
            if cls_accessory:
                accessory = prediction[0]['accessories'][index].detach().cpu()
                result['accessory_score'] = accessory
                result['accessory_gt'] = int(label in ACCESSORIES_ID_LIST)
            predicted_results.append(result)
    result = {}
    if cls_accessory:
        accessory = target[0]['accessories'].detach().cpu()
        targets.append({"accessories":accessory})
    labels = [CATEGORY_LABELS[index-1] for index in target[0]['labels'].detach().cpu()]
    targets.append({"labels":labels})
    return predicted_results#, targets

def predicted_bbox(image, prediction, classes):
    bbox_img = np.array(to_pil(torch.zeros_like(image))).copy()
    real_bbox_img = np.array(to_pil(torch.zeros_like(image))).copy()
    for index, element in enumerate(prediction[0]['boxes']):
        box=element.detach().cpu()
        scr=prediction[0]['scores'][index].detach().cpu()
        label = classes[prediction[0]['labels'][index].detach().cpu()]
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
    images = [image, torch.from_numpy(bbox_img).permute(2,0,1)]
      
    return [image.permute(1,2,0).numpy(), bbox_img]

def visualize_bbox(image, prediction, target, classes):
    image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).byte()
    bbox_img = np.array(to_pil(torch.zeros_like(image))).copy()
    real_bbox_img = np.array(to_pil(torch.zeros_like(image))).copy()
    for index, element in enumerate(prediction[0]['boxes']):
        box=element.detach().cpu()
        scr=prediction[0]['scores'][index].detach().cpu()
        label = classes[prediction[0]['labels'][index].detach().cpu()]
        if scr>0.5:
            print(box)
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


    for index, element in enumerate(prediction[0]['boxes']):
        msk=prediction[0]['masks'][index,0].detach().cpu()
        scr=prediction[0]['scores'][index].detach().cpu()
        label = prediction[0]['labels'][index].detach().cpu()
        if scr>0.5:
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


