import sys

sys.path.append("../../../../../")
from fedlab.utils.functional import save_dict
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

import torchvision

root = '../../../../datasets/data/mnist/'
trainset = torchvision.datasets.MNIST(root=root, train=True, download=True)

data_indices = noniid_slicing(trainset, num_clients=100, num_shards=200)
save_dict(data_indices, "mnist_noniid.pkl")

data_indices = random_slicing(trainset, num_clients=100)
save_dict(data_indices, "mnist_iid.pkl")
