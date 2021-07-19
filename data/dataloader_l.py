import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

sys.path.append("../tools")
import dataset as ds
import reproducible

from train_l import hyperparameters


class RoadSegmentationDataModule(pl.LightningDataModule):
    

    def __init__(self):
        super.__init__()


    def prepare_data(self):
        self.dataset = ds.RoadSegmentationDataset()


    def train_dataloader(self):
        # size of dataset needed to compute split
        dataset_size = len(self.dataset) 

        transform = transforms.Compose([
            # we want our data to be stored as tensors
            ds.ToFloatTensor(),
            ds.Normalize(self.dataset),
        ])

        self.dataset.set_transforms(transform)
        
        train_split_size = int(dataset_size * hyperparameters["train_eval_ratio"])

        self.train_dataset, self.eval_dataset = torch.utils.data.random_split(
            self.dataset, 
            [train_split_size, dataset_size - train_split_size])

        train_dataloader = DataLoader(self.train_dataset, 
            batch_size=hyperparameters["batch_size"], 
            shuffle=hyperparameters["shuffle"], 
            worker_init_fn=reproducible.seed_worker if hyperparameters["reproducible"] else None, 
            generator=reproducible.g if hyperparameters["reproducible"] else None)

        return train_dataloader


    def val_dataloader(self):
        return DataLoader(self.eval_dataset, 
            batch_size=hyperparameters["batch_size"], 
            shuffle=hyperparameters["shuffle"], 
            worker_init_fn=reproducible.seed_worker if hyperparameters["reproducible"] else None, 
            generator=reproducible.g if hyperparameters["reproducible"] else None)


    def test_dataloader(self):
        print("not implemented")
        return None