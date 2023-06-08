import os
import numpy as np
import torch
import torch.utils.data
from pycocotools.coco import COCO

# Define the DATASET
import cv2
import numpy as np
import os
import glob as glob
from utils.utils import pil_loader 

# Add the 14th category: accessory if bag, belt, sunglasses, headwear or scarf tie is present.

ACCESSORY_CATEGORY_ID = 14
ACCESSORIES_ID_LIST = [1,2,7,12,13]

class ModaNetDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, ann_file_name, classes, size, args, transforms=None):
        self.dir_path = dir_path
        self.transforms = transforms
        self.classes = classes
        self.ann_file_name = ann_file_name
        self.width, self.height = size
        self.args = args
        
        self.annotation_file_path = os.path.join(dir_path, self.ann_file_name)
        self.annotations = COCO(annotation_file=self.annotation_file_path)

        self.imgs = list(sorted(os.listdir(os.path.join(dir_path, "Images"))))
        self.img_ids = list(sorted(self.annotations.getImgIds()))



    def __getitem__(self, idx):
        # load images ad masks
        #prendo l'immagine con id = index
        img_file_name = self.imgs[idx]
        id = img_file_name.lstrip('0')
        id = id[:-4]
        img_id = int(id)
        img_path = os.path.join(self.dir_path, "Images", img_file_name)
        img = pil_loader(img_path)
        
        image_width, image_height = img.size
        
        # carico le ANNOTAZIONI di questa immagine
        ann_ids = self.annotations.getAnnIds(imgIds=[img_id])
        img_anns = self.annotations.loadAnns(ann_ids)
        
        
        boxes = []
        labels = []
        mask = []
        img = img.resize((self.width, self.height))
        if self.args .cls_accessory:
            is_accessory = 0
        for ann in img_anns:
            current_mask = self.annotations.annToMask(ann)
            current_mask= (current_mask > 0).astype(np.uint8)
            current_mask = cv2.resize(current_mask, (self.width, self.height), cv2.INTER_LINEAR)
            mask.append(current_mask)
            xmin = int(ann['bbox'][0])
            ymin = int(ann['bbox'][1])
            xmax = xmin+int(ann['bbox'][2])
            ymax = ymin+int(ann['bbox'][3])
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
            labels.append(ann['category_id'])
            if self.args .cls_accessory:
                if ann['category_id'] in ACCESSORIES_ID_LIST:
                    # accessorio presente
                    is_accessory = 1

        mask = np.array(mask)
        mask = torch.as_tensor(mask,  dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        #print(labels)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels 
        target["masks"] = mask
        image_id = torch.tensor([img_id])
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.args .cls_accessory:
            target["isaccessory"]=torch.tensor([is_accessory])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)



