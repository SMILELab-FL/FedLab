# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from .basic_dataset import FedDataset, BaseDataset
from ...utils.dataset.functional import noniid_slicing, random_slicing

class RotatedCIFAR10(FedDataset):
    """Rotate CIFAR10 and patrition them.

        Args:
            root (str): Path to download raw dataset.
            path (str): Path to save partitioned subdataset.
            num_clients (int): Number of clients.
        """
    def __init__(self, root, save_dir, num_clients):
        self.root = os.path.expanduser(root)
        self.dir = save_dir
        self.num_clients = num_clients
        # "./datasets/rotated_mnist/"
        if os.path.exists(save_dir) is not True:
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))

        self.transform  = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def preprocess(self, shards, thetas = [0, 180]):
        """_summary_

        Args:
            shards (_type_): _description_
            thetas (list, optional): _description_. Defaults to [0, 180].
        """
        cifar10 = torchvision.datasets.CIFAR10(self.root, train=True)
        id = 0
        for theta in thetas:
            rotated_data = []
            partition = random_slicing(cifar10, shards)
            for x, _ in cifar10:
                x = self.transform(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
            for key, value in partition.items():
                data = [rotated_data[i] for i in value]
                label = [cifar10.targets[i] for i in value]
                dataset = BaseDataset(data, label)
                torch.save(dataset, os.path.join(self.dir, "train", "data{}.pkl".format(id)))
                id += 1

        # test
        cifar10_test = torchvision.datasets.CIFAR10(self.root, train=False)
        labels = cifar10_test.targets
        for i, theta in enumerate(thetas):
            rotated_data = []
            for x, y in cifar10_test:
                x = self.transform(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
            dataset = BaseDataset(rotated_data, labels)
            torch.save(dataset, os.path.join(self.dir,"test", "data{}.pkl".format(i)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.dir, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
