import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision.io import ImageReadMode
from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import os

import random

import pytorch_lightning as pl

class RoadSatelliteSet(Dataset):
    def __init__(self, dataset, transform_img=None, transform_tuple=None, random_transform_tuple=None):
        self.dataset = dataset
        
        self.transform_img = transform_img
        self.transform_tuple = transform_tuple
        self.random_transform_tuple = random_transform_tuple

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        
        if self.transform_img:
            img = self.transform_img(img)
            
        if self.random_transform_tuple:
            img, mask = self.random_transform_tuple(img, mask)

        if self.transform_tuple:
            img, mask = self.transform_tuple(img, mask)

        return (img, mask)

class RoadSatelliteModule(pl.LightningDataModule):
    def __init__(self, num_workers=4, batch_size=8):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size

        # Stupid fixy fix
        self.prepare_data()
        self.setup('fit')

    def prepare_data(self):
        self.train_images = self.read_images('train/images/', ImageReadMode.RGB)
                
        self.train_masks = self.read_images('train/groundtruth/', ImageReadMode.GRAY)
    
        for i, train_mask in enumerate(self.train_masks):
            self.train_masks[i][self.train_masks[i] > 0] = 1
            
        self.train_zip = list(zip(self.train_images, self.train_masks))
        
        self.test_images = self.read_images('test/', ImageReadMode.RGB)
        
        self.transforms_img = T.Compose(
            [
                T.RandomEqualize(p=1.0),
                T.GaussianBlur(3, 5),
                T.RandomAdjustSharpness(3, 1),
            ]
        )
        
    def setup(self, stage=None):
        if stage in (None, 'fit'): 
            train_length = int(len(self.train_zip) * 0.8)
            valid_length = len(self.train_zip) - train_length

            self.train_data, self.valid_data = random_split(self.train_zip, [train_length, valid_length])
            
            self.train_dataset = RoadSatelliteSet(self.train_data, self.transforms_img, self.augmentations, self.randomAugmentations)
            self.valid_dataset = RoadSatelliteSet(self.valid_data, self.transforms_img, self.augmentations)
            
        if stage in (None, 'test'):
            self.test_data = self.test_images
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, 
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
         )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )
    
    def read_images(self, data_dir, read_mode):
        return [read_image(data_dir + file, read_mode) for file in os.listdir(data_dir)]
    
    def randomAugmentations(self, img, mask):    
        if random.random() > 0.5:
            angle = T.RandomRotation(degrees=(0, 360)).get_params([0, 360])

            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)

        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
            
        return img, mask
    
    def augmentations(self, img, mask):
        img = self.get_patches_averages_rgb(img)
        mask = self.get_patches_averages_rgb(mask, True)

        return img, mask

        size = (256, 256)    
        
        imgs_crop = T.FiveCrop(size=size)(img)
        masks_crop = T.FiveCrop(size=size)(mask)
        
        chosen_index = random.randint(0, 4)
        
        return imgs_crop[chosen_index], masks_crop[chosen_index]

    def get_patches_from_image(self, img):
        size = 5
        stride = 5
        patches = img.unfold(1, size, stride).unfold(2, size, stride)
        
        return patches

    def get_patches_averages_rgb(self, img, is_mask=False):
        patches = self.get_patches_from_image(img)
        
        patches_avg = patches.float().mean((3, 4))
        
        if is_mask:
            patches_avg[patches_avg > 0.25] = 1.0
            
        return patches_avg.byte()
        