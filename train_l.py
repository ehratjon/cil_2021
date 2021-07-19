import sys
import os
import numpy as np
import torch
import pytorch_lightning as pl


# all code files used are stored in a tools folder
# this allows us to directly import those files
sys.path.append("cil_data")
sys.path.append("data")
sys.path.append("models")
sys.path.append("tools")


# import of local files
import dataloader_l
import simple_models_l
import loss_functions_l
import reproducible
from datawriter import datawriter


# specify hyperparameters
hyperparameters = {
    "epochs": 5,
    "learning_rate": 1e-2, 
    "batch_size": 44, 
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


if(hyperparameters["reproducible"]): reproducible.set_deterministic()    


def main():
    dataloader = dataloader_l.RoadSegmentationDataModule(hyperparameters)
    trainer = pl.Trainer()
    trainer.fit(simple_models_l.OneNodeModel(hyperparameters), dataloader)


if __name__ == "__main__":
    main()