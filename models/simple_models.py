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
        