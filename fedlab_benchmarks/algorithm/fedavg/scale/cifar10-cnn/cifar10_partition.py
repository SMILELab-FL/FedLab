from fedlab.utils.functional import save_dict
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

import torchvision

root = '../../../../datasets/data/cifar10/'
trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=False)

data_indices = noniid_slicing(trainset, num_clients=100, num_shards=200)
save_dict(data_indices, "cifar10_noniid.pkl")

data_indices = random_slicing(trainset, num_clients=100)
save_dict(data_indices, "cifar10_iid.pkl")
