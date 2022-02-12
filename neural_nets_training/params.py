import torch
import torch.nn as nn
from collections import namedtuple

torch.manual_seed(0)

privacy_data = namedtuple("privacy_data",
                          ["privacy_engine", "epsilon", "delta"])

ASSETS_PATH = "."

## global params
batch_size = BATCH_SIZE = 128
MAX_GRAD_NORM = 0.3
max_lr = 1e-3
weight_decay = 1e-4
optimizer_type = torch.optim.Adam
criterion = nn.CrossEntropyLoss()
DELTA = 1e-5

# to do: put these in a nice place


# transfer to GPU
def get_default_device():
    """Pick GPU if available, else CPU"""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device(
        'cpu')


#device = get_default_device()
