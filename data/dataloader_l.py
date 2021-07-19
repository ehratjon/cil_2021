import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

sys.path.append("../tools")
import dataset as ds
import reproducible


class RoadSegmentationDataModule(pl.LightningDataModule):
    

    def __init__(self, hyperparameters=None):
        super(RoadSegmentationDataModule, self).__init__()
        self.hyperparameters = hyperparameters


    def prepare_data(self):
        self.dataset = ds.RoadSegmentationDataset()

        transform = transforms.Compose([
            # we want our data to be stored as tensors
            ds.ToFloatTensor(),
            ds.Normalize(self.dataset),
        ])

        self.dataset.set_transforms(transform)
        
        dataset_size = len(self.dataset) 
        train_split_size = int(dataset_size * self.hyperparameters["train_eval_ratio"])

        self.train_dataset, self.eval_dataset = torch.utils.data.random_split(
            self.dataset, 
            [train_split_size, dataset_size - train_split_size])


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
            batch_size=self.hyperparameters["batch_size"], 
            shuffle=self.hyperparameters["shuffle"], 
            worker_init_fn=reproducible.seed_worker if self.hyperparameters["reproducible"] else None, 
            num_workers=self.hyperparameters["num_workers"],
            generator=reproducible.g if self.hyperparameters["reproducible"] else None)


    def val_dataloader(self):
        return DataLoader(self.eval_dataset, 
            batch_size=self.hyperparameters["batch_size"], 
            shuffle=self.hyperparameters["shuffle_val"], 
            worker_init_fn=reproducible.seed_worker if self.hyperparameters["reproducible"] else None, 
            num_workers=self.hyperparameters["num_workers"],
            generator=reproducible.g if self.hyperparameters["reproducible"] else None)


    def test_dataloader(self):
        print("not implemented")
        return None