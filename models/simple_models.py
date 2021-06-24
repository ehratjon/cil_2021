import torch

from torch import nn

# Returns a tensor with all 0 of the same dimension as input
class ZeroModel(nn.Module):
    def __init__(self):
        super(ZeroModel, self).__init__()

    # defines forward pass (never call yourself)
    def forward(self, x):
        return torch.zeros_like(x)