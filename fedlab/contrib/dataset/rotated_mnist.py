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


class RotatedMNIST(FedDataset):
    """Rotate MNIST and partition them.

        Args:
            root (str): Path to download raw dataset.
            path (str): Path to save partitioned subdataset.
            num_clients (int): Number of clients.
        """
    def __init__(self, root, path, num) -> None:
        self.root = os.path.expanduser(root)
        self.path = path
        self.num = num

    def preprocess(self, thetas=[0, 90, 180, 270], download=True):
        self.download = download
        # "./datasets/rotated_mnist/"
        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

        # train
        mnist = torchvision.datasets.MNIST(self.root,
                                           train=True,
                                           download=self.download,
                                           transform=transforms.ToTensor())
        id = 0
        to_tensor = transforms.ToTensor()
        for theta in thetas:
            rotated_data = []
            labels = []
            partition = random_slicing(mnist, int(self.num / len(thetas)))
            for x, y in mnist:
                x = to_tensor(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
                labels.append(y)
            for key, value in partition.items():
                data = [rotated_data[i] for i in value]
                label = [labels[i] for i in value]
                dataset = BaseDataset(data, label)
                torch.save(
                    dataset,
                    os.path.join(self.dir, "train", "data{}.pkl".format(id)))
                id += 1

        # test
        mnist_test = torchvision.datasets.MNIST(
            self.root,
            train=False,
            download=self.download,
            transform=transforms.ToTensor())
        labels = mnist_test.targets
        for i, theta in enumerate(thetas):
            rotated_data = []
            for x, y in mnist_test:
                x = to_tensor(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
            dataset = BaseDataset(rotated_data, labels)
            torch.save(dataset,
                       os.path.join(self.dir, "test", "data{}.pkl".format(i)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(
            os.path.join(self.dir, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
