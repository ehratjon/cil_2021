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
        # we need to differentiate between batching and no batching
        # without batching we get a 3d tensor (with the 1st dimension being the channels)
        if(len(x.shape) == 3):
            return torch.zeros_like(x[0], dtype=torch.float, requires_grad=False)
        # with batching we get a 4d tensor (with the 1st dimension being the number of samples)
        else:
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
        batching = (len(x.shape) == 4)
        # transpose to get the channels at last index
        x_t = x.permute((0, 2, 3, 1)) if batching else x.permute((1, 2, 0))
        # multiply with weights
        z = torch.matmul(x_t, self.weights)
        # flatten last dimension
        z_flat = torch.flatten(z, start_dim=2) if batching else torch.flatten(z, start_dim=1)
        # relu round

        z_relu = self.relu(z_flat)
        return z_relu
