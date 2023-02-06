from tqdm import tqdm
import torch
from PIL import Image
import utils.transforms as T
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.utils import draw_segmentation_masks, make_grid
import matplotlib.pyplot as plt

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

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        print(img.size)
        return img

to_tensor = T.PILToTensor()
to_pil = transforms.ToPILImage()

def collate_fn(batch):
    return tuple(zip(*batch)) 


# define train and test functions

def train(train_data_loader, model, optimizer, device):
    print('Training')
    train_itr = 0
    train_loss_list = []
    
    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        images =  list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets) # when given images and targets as input it will return the loss
        print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list

def get_train_transform():
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.RandomHorizontalFlip(p=0.5), # in this example this is not needed: why?
    ])
# define the validation transforms
def get_valid_transform():
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])

def validate(valid_data_loader, model, optimizer, device):
    print('Validating')
    val_itr = 0
    val_loss_list = []
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}\n\n")
    return val_loss_list

def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'res/mask_rcnn.pth')