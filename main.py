

from torch.utils.data import DataLoader
import torch
from datasets.modanet import ModaNetDataset
from models.mask_rcnn import Mask_RCNN

from utils.utils import get_train_transform,get_valid_transform, collate_fn
import argparse

from solver import Solver
import os

def list_files(parent_dir):
    file_map = {}  # Initialize an empty dictionary to store the file mappings
    for child_dir in os.listdir(parent_dir):
        child_path = os.path.join(parent_dir, child_dir)
        if os.path.isdir(child_path):
            files = os.listdir(child_path)
            if len(files) > 0:
                print("Files in", child_dir + ":")
                for file in files:
                    file_path = os.path.join(child_path, file)
                    if file == ".DS_STORE":
                        os.remove(file_path)
                    else:
                        print(file)
                        file_map[child_dir] = file  # Add the mapping to the dictionary
                print()
    return file_map 


def get_args():
    parser = argparse.ArgumentParser()   

    parser.add_argument('--run_name', type=str, default="run_1", help='name of current run')
    parser.add_argument('--model_name', type=str, default="first_train", help='name of the model to be saved/loaded')

    parser.add_argument('--annotations_file', type=str, default="modanet2018_instances_train.json", help='name of the annotations file')

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of elements in batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=500, help='print losses every N iteration')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'], help = 'optimizer used for training')

    parser.add_argument('--dataset_path', type=str, default='./ModaNetDatasets', help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./', help='path were to save the trained model')

    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'evaluate', 'debug'], help = 'net mode (train or test)')

    parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained coco weights.')

    parser.add_argument('--version', type=str, default='V1', choices=['V1', 'V2'], help = 'maskrcnn version (V1 or improved V2)')

    parser.add_argument('--cls_accessory', action='store_true', help='Add a binary classifier for the accessories')

    parser.add_argument('--random_seed', type=bool, default=True, help='Use same random seed to get same train/valid/test sets for every training.')

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

    ANN_FILE_NAME = args.annotations_file

    # use our dataset and defined transformations
    total_dataset = ModaNetDataset(
        args.dataset_path, ANN_FILE_NAME, CLASSES, IMAGE_SIZE, args, get_train_transform()
    )
    print(len(total_dataset))

    # split the dataset in train and test set
    if args.random_seed:
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
    solver = Solver(train_loader=data_loader,
            valid_loader=data_loader_valid,
            test_loader=data_loader_test,
            device=DEVICE,
            args=args,
            classes = CLASSES)

    # TRAIN model
    if args.mode == "train":
        solver.train()
    elif args.mode == "test":
        solver.test()
    elif args.mode == "evaluate":
        solver.evaluate(0)
    elif args.mode == "debug":
        solver.debug()
    else:
        raise ValueError("Not valid mode")

if __name__ == "__main__":
    args = get_args()
    print(args)
    #print(list_files(args.checkpoint_path))
    main(args)

