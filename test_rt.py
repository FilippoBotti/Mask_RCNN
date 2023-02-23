
import numpy as np
import cv2 
from models.mask_rcnn import Mask_RCNN
import argparse
import os
import torch
from torchvision import transforms
from utils.utils import to_pil

IMAGE_SIZE=[400,600]
CLASSES = [
    '__background__', '1','2','3','4','5','6','7','8','9','10','11','12','13'
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
    # for box_num in range(len(target['boxes'])):
    #     box = target['boxes'][box_num]
    #     label = classes[target['labels'][box_num]]

    #     cv2.rectangle(
    #         real_bbox_img, 
    #         (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
    #         (0, 255, 0), 2
    #     )
    #     cv2.putText(
    #         real_bbox_img, label, (int(box[0]), int(box[1]-5)), 
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    #     )
    # images = [image, torch.from_numpy(bbox_img).permute(2,0,1), torch.from_numpy(real_bbox_img).permute(2,0,1)]
    return bbox_img

def main(args):
    model = Mask_RCNN(len(CLASSES))
    check_path = os.path.join(args.checkpoint_path, "modanet_maskRCNN_first_train.pth")
    model.load_state_dict(torch.load(check_path, map_location=torch.device('cpu')))
    print("Model loaded!")
    model.eval()



    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
       # frame = np.resize(frame, (IMAGE_SIZE))
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        transform = get_transform()
        frame = transform(frame)
        pred = model([frame])
        images = visualize_bbox(frame,pred,CLASSES)
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.imshow('frame', images)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()










if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
