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


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

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
    # print("asd",(target['masks']>0).size())
    # for i in range(len(target['masks'])):
    #     msk=target['masks'][i,0].detach().cpu().numpy()
    #     scr=target['scores'][i].detach().cpu().numpy()
    #     if scr>0.8 :

    mask = draw_segmentation_masks(original_img,target['masks']>0)
    images = [original_img, torch.from_numpy(img).permute(2,0,1), mask]

    return images

def visualize_result(image, target, classes):
    original_img = image.byte()
    mask=original_img.clone()
    img = np.array(to_pil(image.byte())).copy()
    # for box_num in range(len(target['boxes'])):
    #     box = target['boxes'][box_num]
    #     label = classes[target['labels'][box_num]]
        

    #     cv2.rectangle(
    #         img, 
    #         (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
    #         (0, 255, 0), 2
    #     )
    #     cv2.putText(
    #         img, label, (int(box[0]), int(box[1]-5)), 
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    #     )
    # print("asd",(target['masks']>0).size())
    # for i in range(len(target['masks'])):
    #     msk=target['masks'][i,0].detach().cpu().numpy()
    #     scr=target['scores'][i].detach().cpu().numpy()
    #     if scr>0.8 :
    for el in target:
        print(el.size())
        mask = draw_segmentation_masks(original_img,el.squeeze(1)>0)

    images = [original_img, torch.from_numpy(img).permute(2,0,1), mask]

    return images

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
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'res/mask_rcnn.pth')
    
def get_prediction(pred, classes, confidence=0.8):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0
    
    """
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    print(pred_score.size)
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # print(pred[0]['labels'].numpy().max())
    pred_class = [classes[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class