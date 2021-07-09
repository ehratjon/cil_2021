import torch

from torch import nn

# Returns a tensor with all 0 of the same dimension as input
class ZeroModel(nn.Module):
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
class OneNodeModel(nn.Module):
    def __init__(self):
        super(OneNodeModel, self).__init__()
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
