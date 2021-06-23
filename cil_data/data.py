import os

from skimage import io, transform, img_as_float, img_as_ubyte
import cv2

import torch
from torch.utils.data import Dataset

"""
Author: Jonas Passweg
Most of the code comes from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
This file contains a dataset class for our road segmentdation pictures and transformer classes
that can be applied to the datasets.

Datasets for training data can be accessed like this:
# Initialize dataset:
sys.path.append(__path to cil_data folder__)
import data
dataset = data.RoadSegmentationDataset()
# access image with id 10:
image = dataset[10]["image"]
ground_truth = dataset[10]["ground_truth"]

Datasets for testing data can be accessed like this:
# Initialize dataset:
sys.path.append(__path to cil_data folder__)
import data
dataset = data.RoadSegmentationTestDataset()
# access image with id 10:
image = dataset[10]["image"]

Transformed datasets can be accessed like this:
#TODO

Note:
If the file is run without the cil_data folder being in the same directory, 
you need to set the root_dir to where you cil_data folder is.
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

    def __len__(self):
        return len(os.listdir(self.path_training_images))

    """
    First available item has id 1
    Return: sample = {'image': image, 'ground_truth': ground_truth}
    """
    def __getitem__(self, idx):
        """
        Args:
            idx (number) = id of the image we want to use
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Since all ids in the folder are consectuive, we can just grab the image with id idx.
        image_name        = os.path.join(self.path_training_images, "satImage_" + '%03d' % idx + ".png")
        ground_truth_name = os.path.join(self.path_training_groundtruth, "satImage_" + '%03d' % idx + ".png")
        
        image = io.imread(image_name)
        ground_truth = io.imread(ground_truth_name)

        sample = {'image': image, 'ground_truth': ground_truth}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RoadSegmentationTestDataset(Dataset):
    """Road Semgentation Test dataset."""

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

        self.path_test_images = os.path.join(root_dir, "test_images/test_images/")

    def __len__(self):
        return len(os.listdir(self.path_test_images))

    """
    First available item has id 1
    Return: sample = {'image': image, 'ground_truth': None}
    """
    def __getitem__(self, idx):
        """
        Args:
            idx (number) = id of the image we want to use
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        """
        Here we need to do something a bit different since not all ids are present
        in the folder. We list all files and take the image with idx-1 out (as the 
        first element has id 1 but is indexed with 0).
        """
        image_paths = os.listdir(self.path_test_images)
        image_name = image_paths[idx-1]
        
        image = io.imread(image_name)

        sample = {'image': image, 'ground_truth': None}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Downscale(object):
    """
    Downscale the image in a sample to a given size.
    Uses skimage.resize

    Args:
        output_size (tuple (Height, Width)): Desired output size.
        anti_aliasing: set to False to alias ("=" remove blur)
    """
    def __init__(self, output_size, anti_aliasing=False):
        self.output_size = output_size
        self.anti_aliasing = anti_aliasing

    def __call__(self, sample):
        image, ground_truth = sample['image'], sample['ground_truth']

        # resize image
        new_image = transform.resize(image, self.output_size, anti_aliasing = self.anti_aliasing)

        # resize ground truth if it exists
        if(ground_truth):
            new_ground_truth = transform.resize(ground_truth, self.output_size, 
                anti_aliasing = self.anti_aliasing)
        else:
            new_ground_truth = None

        return {'image': new_image, 'ground_truth': new_ground_truth}


class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    Uses cv2.resize which can both up- and downscale

    Args:
        output_size (tuple): Desired output size: (Height, Width). If tuple, output is
            matched to output_size.
    """
    def __init__(self, output_size):
        # open cv uses BGR instead of RGB and we therefore need to reverse width and height
        self.output_size = tuple(reversed(output_size))

    def __call__(self, sample):
        image, ground_truth = sample['image'], sample['ground_truth']
        
        # convert to open cv image, resize, and back
        # RGB (skimage) vs. BGR (opencv)
        cv_image = img_as_ubyte(image)
        new_cv_image = cv2.resize(cv_image, self.output_size, interpolation = cv2.INTER_AREA)
        new_image = img_as_float(new_cv_image)

        # same for ground truth 
        if(ground_truth):
            cv_ground_truth = img_as_ubyte(ground_truth)
            new_cv_ground_truth = cv2.resize(cv_ground_truth, self.output_size, interpolation = cv2.INTER_AREA)
            new_ground_truth = img_as_float(new_cv_ground_truth)
        else:
            new_ground_truth = None

        return {'image': new_image, 'ground_truth': new_ground_truth}


class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, ground_truth = sample['image'], sample['ground_truth']

        height, width = image.shape[:2]
        new_height, new_width = self.output_size

        top = torch.randint(0, height - new_height)
        left = torch.randint(0, width - new_width)

        new_image = image[top: top + new_height, left: left + new_width]

        if(ground_truth):
            new_ground_truth = ground_truth[top: top + new_height, left: left + new_width]
        else:
            new_ground_truth = None

        return {'image': new_image, 'ground_truth': new_ground_truth}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, ground_truth = sample['image'], sample['ground_truth']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        ground_truth = ground_truth.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(ground_truth)}