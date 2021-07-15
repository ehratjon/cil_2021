import sys
import os
import numpy as np
import torch


# all code files used are stored in a tools folder
# this allows us to directly import those files
sys.path.append("cil_data")
sys.path.append("data")
sys.path.append("models")
sys.path.append("tools")


# import of local files
import dataloader
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
    # choose if params are being stored in results file
    "write_params": True,
}


if(hyperparameters["reproducible"]): reproducible.set_deterministic()


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
    dw = datawriter(device, hyperparameters)
    
    # set up dataloaders
    train_dataloader, eval_dataloader = dataloader.get_dataloader(hyperparameters)
    
    # choose model
    load_model = hyperparameters["load_model"] and os.path.exists("model.pth")
    model = torch.load('model.pth').to(device) if load_model else simple_models.OneNodeModel().to(device)
    dw.set_model(model, load_model)
    
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