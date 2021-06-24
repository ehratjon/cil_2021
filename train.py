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

# specify transforms you want for your data:
data_transform = transforms.Compose([

])

# specify dataset
dataset = data.RoadSegmentationDataset()

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# print model
model = simple_models.SimpleNeuralNetwork().to(device)
print(model)

X = torch.rand(1, 400, 400, device=device)
y_pred = model(X).item()
print(f"Predicted class: {y_pred}")