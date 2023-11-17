from tqdm import tqdm
import torch
from PIL import Image
import utils.transforms as T
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.utils import draw_segmentation_masks, make_grid
import matplotlib.pyplot as plt
from utils.utils import show
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torch
from datasets.modanet import ModaNetDataset
from datasets.fashionpedia import FashionpediaDataset

from utils.utils import get_train_transform, collate_fn, to_pil
import argparse

from solver import Solver



def visualize_mask(image, target):
    image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).byte()
    mask = torch.zeros_like(image)
    real_mask = torch.zeros_like(image)
    for i in range(len(target['segmentation'])):
        real_mask = draw_segmentation_masks(real_mask,target['segmentation'][i]>0,1)
    results = [image, mask, real_mask]
    return results


def visualize_bbox(image, target, classes):
    image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).byte()
    bbox_img = np.array(to_pil(torch.zeros_like(image))).copy()
    real_bbox_img = np.array(to_pil(torch.zeros_like(image))).copy()
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
    plt.imshow(image.permute(1,2,0)) 
    plt.show()
    plt.imshow( torch.from_numpy(real_bbox_img)) 
    plt.show()
    return images




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
    parser.add_argument('--dataset', type=str, default='modanet', choices=['modanet', 'fashionpedia'], help = 'modanet or fashionpedia dataset')
    parser.add_argument('--cls_accessory', action='store_true', help='Add a binary classifier for the accessories')
    parser.add_argument('--change_anchors', action='store_true', help='Change anchors')

    parser.add_argument('--manual_seed', type=bool, default=True, help='Use same random seed to get same train/valid/test sets for every training.')

    parser.add_argument('--coco_evaluation', type=bool, default=False, help='Use evaluate function from coco_eval. Default uses Mean Average Precision from torchvision')

    return parser.parse_args()

def main(args):
    

    BATCH_SIZE = args.batch_size # increase / decrease according to GPU memeory
    NUM_WORKERS = args.workers
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    IMAGE_SIZE=[400,600]

    # classes: 0 index is reserved for background
    CLASSES = [
        '__background__', '1','2','3','4','5','6','7','8','9','10','11','12','13'
    ]

    CLASSES_FASHIONPEDIA = [
    '__background__', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
    '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
    '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46',
    '47', '48', '49', '50', '51', '52', '53'
    ]

    ANN_FILE_NAME = args.annotations_file

    # use our dataset and defined transformations
    if args.dataset == "modanet":
        total_dataset = ModaNetDataset(
            args.dataset_path, ANN_FILE_NAME, CLASSES, IMAGE_SIZE, args, get_train_transform()
        )
    elif args.dataset == "fashionpedia":
        total_dataset = FashionpediaDataset(
            args.dataset_path, ANN_FILE_NAME, CLASSES_FASHIONPEDIA, args, get_train_transform()
        )
    print(len(total_dataset))

    # split the dataset in train and test set
    if args.manual_seed:
        torch.manual_seed(1)
    indices = torch.randperm(len(total_dataset)).tolist()
    dataset = torch.utils.data.Subset(total_dataset, indices[:-9372])
    dataset_valid = torch.utils.data.Subset(total_dataset, indices[-9372:-4686])
    dataset_test = torch.utils.data.Subset(total_dataset, indices[-4686:])

    # define training and validation data loaders
    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=collate_fn)

    data_loader_valid = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn)

    print(len(dataset.indices))
    print(len(dataset_valid.indices))
    print(len(dataset_test.indices))

    print("Device: ", DEVICE)

    # define solver class
    print("Testing", flush=True)
    i=0
    img_count=5
    for data in data_loader_test:
        if(i==img_count):
            break
        images, targets = data
        #results = visualize_mask(images[0],targets[0])
        results = visualize_bbox(images[0],targets[0],CLASSES_FASHIONPEDIA)
        #show(results)
        i+=1
    
if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)


