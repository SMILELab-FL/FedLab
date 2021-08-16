import torch
from torch import nn



model = nn.Sequential(nn.Linear(20,10), nn.Linear(10,5))
optimizer = None
criterion = None


def unittest_dataset():
    pass
