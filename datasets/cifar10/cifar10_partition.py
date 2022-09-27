import sys


from fedlab.utils.functional import save_dict
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

import torchvision

trainset = torchvision.datasets.CIFAR10(root="./", train=True, download=False)

data_indices = noniid_slicing(trainset, num_clients=10, num_shards=1000)
save_dict(data_indices, "cifar10_noniid.pkl")

data_indices = random_slicing(trainset, num_clients=10)
save_dict(data_indices, "cifar10_iid.pkl")

"""
Please refer to cifar10_partition.ipynb file for usage of CIFAR10Partitioner.

Function ``random_slicing()`` and ``noniid_slicing()`` are deprecated in the future version.
"""
