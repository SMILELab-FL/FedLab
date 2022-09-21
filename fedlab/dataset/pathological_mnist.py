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

from .dataset import FedDataset, BaseDataset
from ..utils.dataset.functional import noniid_slicing, random_slicing


class PathologicalMNIST(FedDataset):
    """_summary_

        Args:
            root (_type_): _description_
            path (_type_): _description_
            num (_type_): _description_
            shards (int, optional): _description_. Defaults to 2.
            download (bool, optional): _description_. Defaults to True.
        """

    def __init__(self, root, path, num=100, shards=200, download=True, preprocess=False) -> None:
        self.root = os.path.expanduser(root)
        self.path = path
        self.num = num
        self.shards = shards
        if preprocess:
            self.preprocess(num, shards, download)

    def preprocess(self, num, shards, download=True):
        self.num = num
        self.shards = shards
        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))
        # train
        mnist = torchvision.datasets.MNIST(self.root, train=True, download=self.download,
                                           transform=transforms.ToTensor())
        data_indices = noniid_slicing(mnist, self.num, self.shards)

        samples, labels = [], []
        for x, y in mnist:
            samples.append(x)
            labels.append(y)
        for id, indices in data_indices.items():
            data, label = [], []
            for idx in indices:
                x, y = samples[idx], labels[idx]
                data.append(x)
                label.append(y)
            dataset = BaseDataset(data, label)
            torch.save(dataset, os.path.join(self.path, "train", "data{}.pkl".format(id)))

    def get_dataset(self, id, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
            cid (int): client id
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        dataset = torch.load(os.path.join(self.path, type, "data{}.pkl".format(id)))
        return dataset

    def get_dataloader(self, id, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
