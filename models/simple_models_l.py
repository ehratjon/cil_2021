import torch
from torch import nn
import pytorch_lightning as pl


"""
For lightning modules, use pl.LightningModule instead of nn.Module
"""

# Returns a tensor with all 0 of the same dimension as input
class ZeroModel(pl.LightningModule):
    def __init__(self):
        super(ZeroModel, self).__init__()
        # optimizer will not work without parameters
        # NOTE: this self.register_parameter seemed not to work for other models
        #       instead, torch.nn.Parameter(...) was needed
        self.register_parameter(name='arbitrary', param=torch.nn.Parameter(torch.randn(3)))

    # defines forward pass (never call yourself)
    def forward(self, x):
        return torch.zeros_like(x[:,0,...], dtype=torch.float, requires_grad=False)
        

"""
one node module with one weight per channel
- multiplies weight with pixel and adds result of channels together
- finish with relu
"""
class OneNodeModel(pl.LightningModule):
    def __init__(self, hyperparameters):
        super(OneNodeModel, self).__init__()
        self.hyperparameters = hyperparameters
        # define single weight
        self.weights = torch.nn.Parameter(torch.randn((3, 1), dtype=torch.float, requires_grad=True))
        self.relu = torch.nn.ReLU()


    # defines forward pass (never call yourself)
    def forward(self, x):
        # transpose to get the channels at last index (first index is size of batch)
        x_t = x.permute((0, 2, 3, 1))
        # multiply with weights
        z = torch.matmul(x_t, self.weights)
        # flatten last dimension
        z_flat = torch.flatten(z, start_dim=2)
        # relu round
        z_relu = self.relu(z_flat)
        return z_relu

    def configure_optimizers(self):
        self.loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), 
            self.hyperparameters["learning_rate"])
        return optimizer


    def training_step(self, train_batch, batch_idx):
        image = train_batch["image"]
        pred = self.forward(image)
        loss = self.loss_fn(pred, train_batch["ground_truth"])
        return loss

    
    def validation_step(self, eval_batch, batch_idx):
        image = eval_batch["image"]
        pred = self.forward(image)
        loss = self.loss_fn(pred, eval_batch["ground_truth"])
        return loss