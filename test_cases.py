import sys
import os

from PIL import Image

import numpy as np
import torch
import random

from torch.utils.data import DataLoader
from torchvision import transforms

# all code files used are stored in a tools folder
# this allows us to directly import those files
sys.path.append("cil_data")
sys.path.append("models")
sys.path.append("tools")

# import of local files
import data
import loss_functions


def test_dataset():
    print("TEST DATASET")
    data_transform = transforms.Compose([data.ToTensor()])
    dataset = data.RoadSegmentationDataset(transform=data_transform)
    
    dataset_size = len(dataset)
    train_split_size = int(dataset_size * 0.9)
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_split_size, dataset_size - train_split_size])

    print("example from train split:")
    sample = train_dataset[0]
    image_tensor = sample["image"]
    ground_truth_tensor = sample["ground_truth"]

    image_path = 'temp_image.png'
    groud_truth_path = 'temp_ground_truth.png'

    image_as_array = image_tensor.numpy()
    image_as_array = image_as_array.transpose((1, 2, 0))
    image = Image.fromarray(image_as_array, 'RGB')
    image.save(image_path)

    ground_truth_as_array = ground_truth_tensor.numpy()
    ground_truth_image = Image.fromarray(ground_truth_as_array)
    ground_truth_image.save(groud_truth_path)

    print("whole data  size:            {}".format(dataset_size))
    print("train split size:            {}".format(len(train_dataset)))
    print("eval  split size:            {}".format(len(eval_dataset)))
    print("image size:                  {}".format(image_tensor.shape))
    print("ground truth size:           {}".format(ground_truth_tensor.shape))
    print("random sample (rnd from random_split) is stored under:")
    print("image is stored under:       {}".format(image_path))
    print("groud_truth is stored under: {}".format(groud_truth_path))
    

def test_dataloader():
    print("TEST DATALOADER")
    data_transform = transforms.Compose([data.ToTensor()])
    dataset = data.RoadSegmentationDataset(transform=data_transform)

    batch_size=15
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset_size = len(dataset)
    dataloader_size = len(dataloader.dataset)
    
    print(f"dataset size:    {dataset_size}")
    print(f"dataloader size: {dataloader_size}")
    print(f"batch size:      {batch_size}")
    for batch_number, sample in enumerate(dataloader):
        samples = sample["original_id"]
        samples_size = len(samples)
        print(f"batch_number: {batch_number:>5d} samples_size: {samples_size:>5d} samples: {str(samples)}")


def test_model():
    print("TEST MODELS")
    print("not implemented")


def test_loss_function():
    print("TEST LOSS FUNCTION")
    print("TEST mean_f_score")
    batch_size = 40
    width = 400
    height = 400

    zeros = torch.zeros((batch_size,height,width), dtype=torch.float)
    ones = torch.ones((batch_size,height,width), dtype=torch.float)

    loss1 = loss_functions.mean_f_score(zeros, zeros)
    loss2 = loss_functions.mean_f_score(zeros, ones)
    print(loss1.item())
    print(loss2.item())
        


# goes through all tests (new ones need to be added manually)
def test_all():
    test_dataset()
    test_dataloader()
    test_model()
    test_loss_function()
    

if __name__ == "__main__":
    test_all()