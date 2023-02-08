

from torch.utils.data import DataLoader
import torch
from datasets.modanet import ModaNetDataset
from models.mask_rcnn import Mask_RCNN

from utils.utils import get_train_transform,get_valid_transform, collate_fn
import argparse
from torch.utils.tensorboard import SummaryWriter

from solver import Solver

def get_args():
    parser = argparse.ArgumentParser()   

    parser.add_argument('--run_name', type=str, default="run_1", help='name of current run')
    parser.add_argument('--model_name', type=str, default="first_train", help='name of the model to be saved/loaded')

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='number of elements in batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=500, help='print losses every N iteration')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'], help = 'optimizer used for training')

    parser.add_argument('--dataset_path', type=str, default='./ModaNetDatasets', help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./', help='path were to save the trained model')

    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')

    return parser.parse_args()

def main(args):
    writer = SummaryWriter('./runs/' + args.run_name + args.opt)

    BATCH_SIZE = args.batch_size # increase / decrease according to GPU memeory
    NUM_WORKERS = args.workers
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    IMAGE_SIZE=[256,256]

    # classes: 0 index is reserved for background
    CLASSES = [
        '__background__', '1','2','3','4','5','6','7','8','9','10','11','12','13'
    ]

    ANN_FILE_NAME = "modanet2018_instances_train.json"
    # location to save model and plots
    OUT_DIR = args.checkpoint_path

    # use our dataset and defined transformations
    dataset = ModaNetDataset(
        args.dataset_path, ANN_FILE_NAME, CLASSES, IMAGE_SIZE, get_train_transform()
    )
    dataset_test = ModaNetDataset(
        args.dataset_path, ANN_FILE_NAME, CLASSES, IMAGE_SIZE, get_valid_transform()
    )
    print(len(dataset))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-9372])
    dataset_valid = torch.utils.data.Subset(dataset_test, indices[-9372:-4686])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-4686:])

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

    # model = Mask_RCNN(NUM_CLASSES)
    # print(model)
    # model = model.to(DEVICE)
    # # construct an optimizer
    # # params = [p for p in model.parameters() if p.requires_grad]
    # # optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)

    #device = torch.device("cuda")
    print("Device: ", DEVICE)

    # define solver class
    solver = Solver(train_loader=data_loader,
            valid_loader=data_loader_valid,
            device=DEVICE,
            writer=writer,
            args=args,
            classes = CLASSES)

    # TRAIN model
    solver.train()

if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
