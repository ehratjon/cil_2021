import os

import pandas as pd
from skimage import io

import torch
from torch.utils.data import Dataset

"""
Dataset can be accessed like this:

# Initialize dataset:
sys.path.append("cil_data")
import data
dataset = data.RoadSegmentationDataset()

# access image with id 10:
image = dataset[10]["image"]
ground_truth = dataset[10]["ground_truth"]
"""

class RoadSegmentationDataset(Dataset):
    """Road Semgentation dataset."""

    def __init__(self, csv_file="", root_dir="cil_data/", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform

        self.path_training_images       = os.path.join(root_dir, "training/training/images/")
        self.path_training_groundtruth  = os.path.join(root_dir, "training/training/groundtruth/")
        self.path_test_images           = os.path.join(root_dir, "test_images/test_images/")

    def __len__(self):
        return len(os.listdir(self.path_training_images))

    def __getitem__(self, idx):
        """
        Args:
            idx (number) = id of the image we want to use
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name          = os.path.join(self.path_training_images, "satImage_" + '%03d' % idx + ".png")
        ground_truth_name = os.path.join(self.path_training_groundtruth, "satImage_" + '%03d' % idx + ".png")
        
        image = io.imread(image_name)
        ground_truth = io.imread(ground_truth_name)

        sample = {'image': image, 'ground_truth': ground_truth}

        if self.transform:
            sample = self.transform(sample)

        return sample