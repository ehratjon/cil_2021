import torch
import random
import numpy as np

# initiate generator to remove randomness from data loader
g = torch.Generator()
g.manual_seed(0)

"""
set random seeds
see: https://pytorch.org/docs/stable/notes/randomness.html
"""
def set_deterministic():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    np.random.seed(0)


"""
For the data loader we will need to additionally set the seed every time
(should only be called if reproducible is set to True)
"""
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)