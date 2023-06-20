
import numpy as np
import cv2 
from models.mask_rcnn import Mask_RCNN
import argparse
import os
import torch
from torchvision import transforms
from utils.utils import to_pil
from torchvision.utils import draw_segmentation_masks, make_grid

IMAGE_SIZE=[400,600]
CLASSES = [
    '__background__', '1','2','3','4','5','6','7','8','9','10','11','12','13'
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
def get_args():
    parser = argparse.ArgumentParser()   
    parser.add_argument('--checkpoint_path', type=str, default='./', help='path were to save the trained model')
    return parser.parse_args()

def get_transform():
        return transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.ConvertImageDtype(torch.float)])
    
def visualize_bbox(image, prediction, classes):
    image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).byte()
    bbox_img = np.array(to_pil(image)).copy()
    print(prediction)
    for i in range(len(prediction[0]['boxes'])):
        box=prediction[0]['boxes'][i].detach().cpu()
        scr=prediction[0]['scores'][i].detach().cpu()
        label = classes[prediction[0]['labels'][i].detach().cpu()]
        if scr > 0.7:
            cv2.rectangle(
            bbox_img, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 2
        )

            cv2.putText(
            bbox_img, CATEGORY_LABELS[int(label)-1], (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2
        )
    return bbox_img

def visualize_mask(image, prediction):
    image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).byte()
    mask = torch.zeros_like(image)
    real_mask = torch.zeros_like(image)


    for index, element in enumerate(prediction[0]['boxes']):
        msk=prediction[0]['masks'][index,0].detach().cpu()
        scr=prediction[0]['scores'][index].detach().cpu()
        label = prediction[0]['labels'][index].detach().cpu()
        if scr>0.7:
            mask = draw_segmentation_masks(mask,msk>0.5,1, MASK_COLORS[label-1])
    mask = mask.cpu().numpy().transpose(1,2,0)
   
    return mask

def get_args():
    parser = argparse.ArgumentParser()   

    parser.add_argument('--model_name', type=str, default="first_train", help='name of the model to be saved/loaded')
    parser.add_argument('--annotations_file', type=str, default="modanet2018_instances_train.json", help='name of the annotations file')

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of elements in batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=500, help='print losses every N iteration')

    parser.add_argument('--trainable_backbone_layers', type=int, default=-1, help='number of trainable (not frozen) layers starting from final block.')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'], help = 'optimizer used for training')

    parser.add_argument('--dataset_path', type=str, default='./ModaNetDatasets', help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./', help='path where to save the trained model')

    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'evaluate', 'debug'], help = 'net mode (train or test)')
    parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained coco weights.')
    parser.add_argument('--version', type=str, default='V1', choices=['V1', 'V2'], help = 'maskrcnn version (V1 or improved V2)')
    parser.add_argument('--cls_accessory', action='store_true', help='Add a binary classifier for the accessories')

    parser.add_argument('--manual_seed', type=bool, default=True, help='Use same random seed to get same train/valid/test sets for every training.')

    parser.add_argument('--coco_evaluation', type=bool, default=False, help='Use evaluate function from coco_eval. Default uses Mean Average Precision from torchvision')

    return parser.parse_args()

def main(args):
    model = Mask_RCNN(len(CLASSES), args=args)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device('cpu')))
    model.eval()



    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i = -1
    
    while True:
        ret, frame = cap.read()
       # frame = np.resize(frame, (IMAGE_SIZE))
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        i+=1
        if i % 60 == 0:
            transform = get_transform()
            frame = transform(frame)
            pred = model([frame])
            mask = visualize_mask(frame,pred)
            bbox = visualize_bbox(frame,pred,CLASSES)
            cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
            cv2.imshow('mask', mask)
            cv2.namedWindow('bbox',cv2.WINDOW_NORMAL)
            cv2.imshow('bbox', bbox)
        else:
            cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break











if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
