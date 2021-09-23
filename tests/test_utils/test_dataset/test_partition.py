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
import os
import numpy as np
from fedlab.utils.dataset.partition import DataPartitioner, CIFAR10Partitioner


class DataPartitionerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_samples = 10000
        cls.num_classes = 10
        cls.num_clients = 100

    def setUp(self) -> None:
        np.random.seed(2021)

    def test_len(self):
        pass

    def test_item(self):
        pass


class CIFAR10PartitionerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_samples = 10000
        cls.num_classes = 10
        cls.num_clients = 100

    def setUp(self) -> None:
        np.random.seed(2021)

    def test_balance_iid(self):
        pass

    def test_unbalance_iid(self):
        pass

    def test_balance_dir(self):
        pass

    def test_unbalance_dir(self):
        pass

    def test_hetero_dir(self):
        pass

    def test_shards(self):
        pass

    def test_len(self):
        pass

    def test_item(self):
        pass
