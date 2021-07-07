import numpy as np
import kaggle
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

# all code files used are stored in a tools folder
# this allows us to directly import those files
import sys
sys.path.append("cil_data")
sys.path.append("models")
sys.path.append("tools")

import data
import simple_models

# check if cuda available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# specify dataset
dataset = data.RoadSegmentationDataset()

print(dataset[0]["image"].shape)
print(dataset[0]["ground_truth"].shape)