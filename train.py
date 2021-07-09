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

# specify if you want your result to be reproducible or not
reproducible = True

# specify if you want to store and load the model
load_model = False
store_model = True

# set random seeds
# see: https://pytorch.org/docs/stable/notes/randomness.html
if(reproducible):
    load_model = False # if model is loaded and stored every time, obviously the results will change
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    np.random.seed(0)


"""
For the data loader we will need to additionally set the seed every time
"""
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


"""
trains ...
"""
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch_number, sample in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(sample["image"])
        loss = loss_fn(pred, sample["ground_truth"])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"batch number: {batch_number:>3d} loss for this batch: {loss.item():>7f}")


"""
evaluates ...
"""
def evaluate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for sample in dataloader:
            pred = model(sample["image"])
            test_loss += loss_fn(pred, sample["ground_truth"]).item()

    test_loss /= num_batches
    print(f"Eval avg loss: {test_loss:>8f} \n")


def main():
    # check if cuda available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # specify hyperparameters
    hyperparameters = {
        "epochs": 5,
        "learning_rate": 1e-3, 
        "batch_size": 44, 
        "shuffle": True,
        "train_eval_ratio": 0.9
    }

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
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [
        train_split_size, 
        dataset_size - train_split_size
        ])

    if(reproducible):
        # initiate generator to remove randomness from data loader
        g = torch.Generator()
        g.manual_seed(0)

        # initiate dataloader for training and evaluation datasets
        train_dataloader = DataLoader(train_dataset, 
            batch_size=hyperparameters["batch_size"], shuffle=hyperparameters["shuffle"], worker_init_fn=seed_worker, generator=g)
        eval_dataloader = DataLoader(eval_dataset, 
            batch_size=hyperparameters["batch_size"], shuffle=hyperparameters["shuffle"], worker_init_fn=seed_worker, generator=g)
    else:
        # initiate dataloader for training and evaluation datasets
        train_dataloader = DataLoader(train_dataset, 
            batch_size=hyperparameters["batch_size"], shuffle=hyperparameters["shuffle"])
        eval_dataloader = DataLoader(eval_dataset, 
            batch_size=hyperparameters["batch_size"], shuffle=hyperparameters["shuffle"])

    # choose model
    if(load_model and os.path.exists("model.pth")):
        print("Model loaded from 'model.pth'")
        model = torch.load('model.pth')
    else:
        model = simple_models.OneNodeModel().to(device)
    
    print("Model used: {} \n" .format(model))

    # choose loss function
    # f1_score with average="samples" is the loss function used for testing
    loss_fn = torch.nn.MSELoss()

    # choose optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=hyperparameters["learning_rate"])

    # train
    for epoch in range(hyperparameters["epochs"]):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        evaluate(eval_dataloader, model, loss_fn)
    
    # stores model such that it can be reloaded later
    if(store_model):
        torch.save(model, 'model.pth')
    
    print("Done!")


if __name__ == "__main__":
    main()