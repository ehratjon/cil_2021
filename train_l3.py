from data import RoadSatelliteModule
from system_l2 import SemanticSegmentationSystem
from model_l2 import NestedUNet
from model_l2 import UNet

import numpy as np
import matplotlib.pyplot as plt
import random

import torchvision
import torchvision.transforms.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import torch

from torchinfo import summary

# specify hyperparameters
hyperparameters = {
    "epochs": 5,
    "learning_rate": 1e-2, 
    # was reduced to 8 so that it can still fit on a gpu
    "batch_size": 8, 
    "shuffle": True,
    # Recommended not to shuffle validation data loader
    "shuffle_val": False, 
    "train_eval_ratio": 0.9,
    "num_workers": 24, # best to set to number of cpus on machine
    # not really hyperparameters but used to set behaviour of model
    "reproducible": True,
    "load_model": False,
    "store_model": True,
    # choose if params/images are being stored in results file
    # these parameters are given to datawriter.py
    # info: stores all parameters in csv files
    # info: stores last batch of evaluation images for each epoch
    "write_params": True,
    "write_images": True,
}

def main():
    pl.seed_everything(42, workers=True)
    
    # prepare data
    road_data = RoadSatelliteModule()
    X, _ = next(iter(road_data.train_dataloader()))
    
    # set model and system
    model = NestedUNet(1, 3).cuda()
    summary(model, input_size=(X.shape))
    system = SemanticSegmentationSystem(model, road_data)

    # choose an early callback function
    early_stop_callback = EarlyStopping(
        monitor='validation_metric',
        patience=20,
        verbose=2,
        mode='max'
    )

    # set up trainer
    trainer = pl.Trainer(
        #fast_dev_run=True,
        gpus=-1,
        auto_select_gpus=True,
        #auto_lr_find=True,
        auto_scale_batch_size='binsearch',
        stochastic_weight_avg=True,
        benchmark=True,
        callbacks=[early_stop_callback]
    )

    # tune system (set parameters)
    trainer.tune(system)

    # fit system (actual training)
    trainer.fit(system)

    # visualize results
    system.visualize_results()

if __name__ == "__main__":
    main()
    print("Done!")