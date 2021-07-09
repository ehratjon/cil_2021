import torch

from torch import nn

# Returns a tensor with all 0 of the same dimension as input
class ZeroModel(nn.Module):
    def __init__(self):
        super(ZeroModel, self).__init__()
        # optimizer will not work without parameters
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
        

# multiplies every pixel with one value
class OneNodeModel(nn.Module):
    def __init__(self):
        super(OneNodeModel, self).__init__()
        # define single weight
        self.weight = torch.randn((3, 1), dtype=torch.float, requires_grad=True)

    # defines forward pass (never call yourself)
    def forward(self, x):
        batching = (len(x.shape) == 4)
        # transpose to get the channels at last index
        x_t = x.transpose(0, 2, 3, 1) if batching else x.transpose(1, 2, 0)
        # multiply with weights
        z = torch.matmul(x_t, self.weight)
        # flatten last dimension
        z_flat = torch.flatten(z, start_dim=2) if batching else torch.flatten(z, start_dim=1)
        # relu round
        z_relu = torch.nn.ReLU(z_flat)
        return z_relu
