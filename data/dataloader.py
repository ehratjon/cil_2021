import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


sys.path.append("../tools")
import dataset as ds
import reproducible


"""
Returns dataloader for training and evaluating model
"""
def get_dataloader(hyperparameters):
    # specify transforms you want for your data:
    data_transform = transforms.Compose([
        # we want our data to be stored as tensors
        ds.ToFloatTensor()
    ])

    # specify dataset
    dataset = ds.RoadSegmentationDataset(transform=data_transform)
    dataset_size = len(dataset) # size of dataset needed to compute split
    train_split_size = int(dataset_size * hyperparameters["train_eval_ratio"])
    # split dataset in training and evaluation sets
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, 
        [train_split_size, dataset_size - train_split_size])

    # initiate dataloader for training and evaluation datasets
    train_dataloader = DataLoader(train_dataset, 
        batch_size=hyperparameters["batch_size"], shuffle=hyperparameters["shuffle"], 
        worker_init_fn=reproducible.seed_worker if hyperparameters["reproducible"] else None, 
        generator=reproducible.g if hyperparameters["reproducible"] else None)
    eval_dataloader = DataLoader(eval_dataset, 
        batch_size=1, shuffle=hyperparameters["shuffle"], 
        worker_init_fn=reproducible.seed_worker if hyperparameters["reproducible"] else None, 
        generator=reproducible.g if hyperparameters["reproducible"] else None)

    return train_dataloader, eval_dataloader