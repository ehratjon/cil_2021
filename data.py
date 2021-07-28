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

from skimage import data, segmentation, filters, color
from skimage.future import graph

#from fast_slic.avx2 import SlicAvx2
import cv2

import matplotlib.pyplot as plt
from skimage.measure import regionprops

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
        self.setup()

    def prepare_data(self):
        self.train_images = self.read_images('train/images/', ImageReadMode.RGB)
                
        self.train_masks = self.read_images('train/groundtruth/', ImageReadMode.GRAY)
    
        for i, train_mask in enumerate(self.train_masks):
            self.train_masks[i][self.train_masks[i] > 0] = 1
            
        self.train_zip = list(zip(self.train_images, self.train_masks))
        
        test_files = [(x[0], x[2]) for x in os.walk('test/')]
        self.file_names = test_files[0][1]
        self.test_images = list()
        for file_name in self.file_names:
            self.test_images.append(read_image(str('test/' + file_name), ImageReadMode.RGB))

        self.transforms_img = T.Compose(
            [
                #T.RandomEqualize(p=1.0),
                #T.GaussianBlur(3, 5),
                #T.RandomAdjustSharpness(3, 1),
            ]
        )
        
        
    def setup(self, stage=None):
        train_length = int(len(self.train_zip) * 0.8)
        valid_length = len(self.train_zip) - train_length

        self.train_data, self.valid_data = random_split(self.train_zip, [train_length, valid_length])

        self.train_dataset = RoadSatelliteSet(self.train_data, self.transforms_img, self.augmentations, self.randomAugmentations)

        self.valid_dataset = RoadSatelliteSet(self.valid_data, self.transforms_img, self.augmentations)

        self.test_data = list(zip(self.test_images, self.file_names))
        self.test_dataset = RoadSatelliteSet(self.test_data, self.transforms_img, self.test_augmentations)

    
    def read_images(self, data_dir, read_mode):
        return [read_image(data_dir + file, read_mode) for file in os.listdir(data_dir)]
    
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
            self.test_dataset, 
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )

    def augmentations(self, img, mask, stage=None):
        img_patches = self.split_image(img, kernel_size=320, stride=40)
        mask_patches = self.split_image(mask, kernel_size=320, stride=40)
            
        index_chosen = random.randint(0, img_patches.shape[0] - 1)
        img, mask = img_patches[index_chosen], mask_patches[index_chosen]
                
        size=stride=2
        img = self.get_patches_averages_rgb(img, size=size, stride=stride)
        mask = self.get_patches_averages_rgb(mask, is_mask=True, size=size, stride=stride)
        
        #img = self.merged_img_rag(img, num_components=2000, compactness=10, thresh=0.03)

        return img, mask

    def test_augmentations(self, img, name):        
        img_patches = self.split_image(img, kernel_size=320, stride=80)
                                
        patches_avg = []
        for patch in img_patches:
            size=stride=2
            patch = self.get_patches_averages_rgb(patch, size=size, stride=stride)
            patch = self.merged_img_rag(patch, num_components=2000, compactness=10, thresh=0.03)
            
            patches_avg.append(patch)

        return torch.stack(patches_avg), name

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
    
    def color_transform(self, img, lower=(0, 0, 0), upper= (60, 50, 200)):
        x = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(x, lower, upper)
        result = cv2.bitwise_and(x, x, mask=mask)

        return torch.from_numpy(cv2.cvtColor(result, cv2.COLOR_HSV2RGB)).permute(2, 0, 1) 
    
    def get_patches_from_image(self, img, size=5, stride=5):
        patches = img.unfold(1, size, stride).unfold(2, size, stride)
        
        return patches

    def get_patches_averages_rgb(self, img, is_mask=False, size=5, stride=5):
        patches = self.get_patches_from_image(img, size, stride)
        
        patches_avg = patches.float().mean((3, 4))
        
        if is_mask:
            patches_avg[patches_avg > 0.25] = 1.0
            
        return patches_avg.byte()

    def weight_boundary(self, graph, src, dst, n):
        """
        Handle merging of nodes of a region boundary region adjacency graph.

        This function computes the `"weight"` and the count `"count"`
        attributes of the edge between `n` and the node formed after
        merging `src` and `dst`.


        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbor of `src` or `dst` or both.

        Returns
        -------
        data : dict
            A dictionary with the "weight" and "count" attributes to be
            assigned for the merged node.

        """
        default = {'weight': 0.0, 'count': 0}

        count_src = graph[src].get(n, default)['count']
        count_dst = graph[dst].get(n, default)['count']

        weight_src = graph[src].get(n, default)['weight']
        weight_dst = graph[dst].get(n, default)['weight']

        count = count_src + count_dst
        return {
            'count': count,
            'weight': (count_src * weight_src + count_dst * weight_dst)/count
        }

    def merge_boundary(self, graph, src, dst):
        """Call back called before merging 2 nodes.

        In this case we don't need to do any computation here.
        """
        pass
    
    def slic_img(self, img):
        img = img.permute(1, 2, 0).numpy()

        slic = SlicAvx2(num_components=4000, compactness=10)
        img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        labels1 = slic.iterate(img_cv2)
        
        out = color.label2rgb(labels1, img, kind='avg', bg_label=0)
        
        return torch.from_numpy(out).permute(2, 0, 1).byte()  
    
    def merged_img_rag(self, img, num_components=1000, compactness=10, thresh=0.05):
        img = img.permute(1, 2, 0).numpy()

        edges = filters.sobel(color.rgb2gray(img))
        
        slic = SlicAvx2(num_components=num_components, compactness=compactness)
        img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        labels = slic.iterate(img_cv2)
              
        g = graph.rag_boundary(labels, edges)

        labels2 = graph.merge_hierarchical(labels, g, thresh=thresh, rag_copy=False,
                                        in_place_merge=True,
                                        merge_func=self.merge_boundary,
                                        weight_func=self.weight_boundary)
        
        out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
        
        return torch.from_numpy(out).permute(2, 0, 1).byte() 
    
    def split_image(self, imgs, kernel_size=240, stride=80):
        if len(imgs.shape) < 4:
            imgs = imgs[None, :, :, :]
        
        B, C, H, W = imgs.shape
        
        patches = imgs.float().unfold(3, kernel_size, stride).unfold(2, kernel_size, stride).permute(0,1,2,3,5,4)
        
        return patches.contiguous().view(patches.shape[0], patches.shape[1], patches.shape[2] * patches.shape[3], patches.shape[4], patches.shape[5]).squeeze(0).transpose(1, 0).byte()
  
    def restore_image(self, patches, H=400, W=400, stride=80):
        B, C, _, _, kernel_size, _ = patches.shape
            
        patches = patches.contiguous().view(B, C, -1, kernel_size*kernel_size)
        patches = patches.permute(0, 1, 3, 2) 
        patches = patches.contiguous().view(B, C*kernel_size*kernel_size, -1)
            
        output = torch.nn.functional.fold(
            patches, output_size=(H, W), kernel_size=kernel_size, stride=stride)
            
        return output

    def restore_image_mask(self, patches, H=400, W=400, stride=80, num_patches_v=3, num_patches_h=3):
        patches = patches.transpose(1, 0).view(patches.shape[1], num_patches_v, num_patches_h, patches.shape[2], patches.shape[3]).unsqueeze(0)    

        B, C, _, _, kernel_size, _ = patches.shape
        
        restored_output = self.restore_image(patches, H, W, stride)
        
        ones = torch.ones((B, C, H, W), device='cuda')
        
        patches_ones = self.split_image(ones, kernel_size, stride).float()
        
        patches_ones = patches_ones.transpose(1, 0).view(patches_ones.shape[1], num_patches_v, num_patches_h, patches_ones.shape[2], patches_ones.shape[3]).unsqueeze(0)    
        
        restored_ones = self.restore_image(patches_ones, H, W, stride)
            
        return restored_output / restored_ones 