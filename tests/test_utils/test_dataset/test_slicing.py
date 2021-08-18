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


import unittest
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing, divide_dataset


class SliceTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls) -> None:
        cls.trainset = torchvision.datasets.MNIST(
            root="./tests/data/mnist/", train=True, download=True, transform=transforms.ToTensor()
        )
        cls.total_client=100

    def noniid_slicing_test(self):
        data_indices = random_slicing(self.trainset, num_clients=self.total_client)

    def random_slicing_test(self):
        data_indices = data_indices = noniid_slicing(
            self.trainset, num_clients=self.total_client, num_shards=200
        )

    def divide_dataset(self):
        pass