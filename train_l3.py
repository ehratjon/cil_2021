import numpy as np
import matplotlib.pyplot as plt
import random
import sys

import torchvision
import torchvision.transforms.functional as F
import torch
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

sys.path.append("cil_data")
sys.path.append("data")
sys.path.append("models")
sys.path.append("tools")

from data_l2 import RoadSatelliteModule
from system_l2 import SemanticSegmentationSystem
from model_l2 import NestedUNet
from model_l2 import UNet


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
        fast_dev_run=True,
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