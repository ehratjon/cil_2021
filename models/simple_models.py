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
        # output is the same size as each channel of the input
        return torch.zeros_like(x[0], requires_grad=False)