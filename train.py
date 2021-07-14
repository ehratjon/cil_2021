import sys
import os

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
import simple_models
import loss_functions
import reproducible
from datawriter import datawriter

# specify hyperparameters
hyperparameters = {
    "epochs": 5,
    "learning_rate": 1e-2, 
    "batch_size": 44, 
    "shuffle": True,
    "train_eval_ratio": 0.9,
    # not really hyperparameters but used to set behaviour of model
    "reproducible": True,
    "load_model": False,
    "store_model": True,
}

if(hyperparameters["reproducible"]): reproducible.set_deterministic()


"""
Returns dataloader for training and evaluating model
"""
def get_dataloader():
    # specify transforms you want for your data:
    data_transform = transforms.Compose([
        # we want our data to be stored as tensors
        data.ToFloatTensor()
    ])

    # specify dataset
    dataset = data.RoadSegmentationDataset(transform=data_transform)
    dataset_size = len(dataset) # size of dataset needed to compute split
    train_split_size = int(dataset_size * hyperparameters["train_eval_ratio"])
    # split dataset in training and evaluation sets
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, 
        [train_split_size, dataset_size - train_split_size])

    # initiate dataloader for training and evaluation datasets
    train_dataloader = DataLoader(train_dataset, 
        batch_size=hyperparameters["batch_size"], shuffle=hyperparameters["shuffle"], 
        worker_init_fn=reproducible.seed_worker if hyperparameters["reproducible"] else None, 
        generator=reproducible.g if hyperparameters["reproducible"] else None)
    eval_dataloader = DataLoader(eval_dataset, 
        batch_size=1, shuffle=hyperparameters["shuffle"], 
        worker_init_fn=reproducible.seed_worker if hyperparameters["reproducible"] else None, 
        generator=reproducible.g if hyperparameters["reproducible"] else None)

    return train_dataloader, eval_dataloader


"""
trains the model using one optimizer step for each batch in the dataloader
"""
def train(dataloader, model, loss_fn, optimizer):
    for _, sample in enumerate(dataloader):
        pred = model(sample["image"])
        loss = loss_fn(pred, sample["ground_truth"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


"""
evaluates the model by computing the average loss over all batches in the data_loader
"""
def evaluate(dataloader, model, loss_fn):
    eval_loss = 0

    with torch.no_grad():
        for sample in dataloader:
            pred = model(sample["image"])
            eval_loss += torch.mean(loss_fn(pred, sample["ground_truth"])).item()

    eval = {
        "eval_loss": eval_loss / len(dataloader),
        "sample": sample,
        "prediction": pred
    }

    return eval


def main():
    # check if cuda available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set up datawriter
    dw = datawriter()
    dw.write_info("Using {} device".format(device))
    dw.write_hyperparameters(hyperparameters)
    
    # set up dataloaders
    train_dataloader, eval_dataloader = get_dataloader()
    
    # choose model
    if(hyperparameters["load_model"] and os.path.exists("model.pth")):
        model = torch.load('model.pth')
        dw.write_info("Model loaded from 'model.pth'")
        dw.set_model(model)
    else:
        model = simple_models.OneNodeModel().to(device)
        dw.set_model(model)
    
    # choose loss function
    loss_fn = torch.nn.MSELoss()

    # choose optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    # train
    for epoch in range(hyperparameters["epochs"]):
        train(train_dataloader, model, loss_fn, optimizer)
        dw.write_eval(epoch, evaluate(eval_dataloader, model, loss_fn))
    
    # stores model such that it can be reloaded later
    if(hyperparameters["store_model"]): torch.save(model, 'model.pth')


if __name__ == "__main__":
    main()
    print("Done!")