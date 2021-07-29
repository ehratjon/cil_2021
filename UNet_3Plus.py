#!/usr/bin/env python
# coding: utf-8

# In[2]:



from data import RoadSatelliteModule
from system import SemanticSegmentationSystem

from models import *

import numpy as np
import matplotlib.pyplot as plt
import random

import torchvision
import torchvision.transforms.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import torch
from torchinfo import summary


# # 1. Preparation

# In[3]:


batch_size = 8
num_workers = 8


# In[4]:


pl.seed_everything(7, workers=True)


# ## 1.1 DataModule

# In[5]:


road_data = RoadSatelliteModule()


# In[6]:


X, y = next(iter(road_data.train_dataloader()))


# # 2. Define Model / System

# In[11]:


model = UNet_3Plus(3, 1)


# In[12]:


system = SemanticSegmentationSystem(model, road_data)


if torch.cuda.is_available():
    gpu_count = -1
    gpu_auto_select = True
    print("GPUs detected.")
    print("There should be ", torch.cuda.device_count(), " GPUs available.")
else:
    gpu_count = 0
    gpu_auto_select = False
    print("No GPU detected.")
    print("Working with CPU")


early_stop_callback = EarlyStopping(
   monitor='validation_accuracy',
   patience=20,
   verbose=2,
   mode='max'
)


trainer = pl.Trainer(
    #fast_dev_run=True,
    gpus=gpu_count,
    auto_select_gpus=gpu_auto_select,
    stochastic_weight_avg=True,
    benchmark=True,
    callbacks=[early_stop_callback]
)


# In[23]:


trainer.fit(system)


if gpu_count != 0:
    model.cuda()
