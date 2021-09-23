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


#
# class DataPartitionerTestCase(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.num_samples = 10000
#         cls.num_classes = 10
#         cls.num_clients = 100
#
#     def setUp(self) -> None:
#         np.random.seed(2021)
#
#     def test_len(self):
#         pass
#
#     def test_item(self):
#         pass


class CIFAR10PartitionerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_classes = 10
        cls.num_clients = 50
        cls.seed = 2021
        targets = []
        for k in range(cls.num_classes):
            targets.append([k for _ in range(5000)])
        cls.num_samples = len(targets)
        targets = np.array(targets)
        cls.targets = targets[np.random.permutation(cls.num_samples)]  # shuffle

    # def setUp(self) -> None:
    #     np.random.seed(self.seed)

    # def test_hetero_dir(self):
    #     # perform partition
    #     hetero_dir_part = CIFAR10Partitioner(self.targets,
    #                                          self.num_clients,
    #                                          balance=None,
    #                                          partition="dirichlet",
    #                                          dir_alpha=0.3,
    #                                          seed=self.seed)

    def test_shards(self):
        pass

    def test_balance_iid(self):
        pass

    def test_unbalance_iid(self):
        pass

    def test_balance_dir(self):
        pass

    def test_unbalance_dir(self):
        pass

    def test_len(self):
        partitioner = CIFAR10Partitioner(self.targets,
                                         self.num_clients,
                                         balance=True,
                                         partition="iid",
                                         seed=self.seed)
        self.assertEqual(len(partitioner), self.num_clients)

    def test_item(self):
        partitioner = CIFAR10Partitioner(self.targets,
                                         self.num_clients,
                                         balance=True,
                                         partition="iid",
                                         seed=self.seed)
        res = [all(partitioner[cid] == partitioner.client_dict[cid]) for cid in
               range(self.num_clients)]
        self.assertTrue(all(res))
