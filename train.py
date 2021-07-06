import numpy as np
import kaggle

import torch
from sklearn.metrics import f1_score

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

        if batch_number % 100 == 0:
            loss, current = loss.item(), batch_number * len(sample["image"])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def evaluate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for sample in dataloader:
            pred = model(sample["image"])
            test_loss += loss_fn(pred, sample["ground_truth"]).item()
            correct += (pred == sample["ground_truth"]).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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
        data.ToTensor()
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

    # initiate dataloader for training and evaluation datasets
    train_dataloader = DataLoader(train_dataset, 
        batch_size=hyperparameters["batch_size"], shuffle=hyperparameters["shuffle"])
    eval_dataloader = DataLoader(eval_dataset, 
        batch_size=hyperparameters["batch_size"], shuffle=hyperparameters["shuffle"])

    # choose model
    model = simple_models.ZeroModel().to(device)

    # choose loss function
    # f1_score with average="samples" is the loss function used for testing
    loss_fn = f1_score

    # choose optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=hyperparameters["learning_rate"])

    # train
    for epoch in range(hyperparameters["epochs"]):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        evaluate(eval_dataloader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    main()