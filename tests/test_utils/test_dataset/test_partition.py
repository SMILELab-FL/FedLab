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
import random
from fedlab.utils.dataset.partition import DataPartitioner, CIFAR10Partitioner, BasicPartitioner, VisionPartitioner, FCUBEPartitioner


class CIFAR10PartitionerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_classes = 10
        cls.num_clients = 50
        cls.seed = 2021
        targets = []
        for k in range(cls.num_classes):
            targets.extend([k for _ in range(500)])
        cls.num_samples = len(targets)
        targets = np.array(targets)
        np.random.seed(cls.seed)
        cls.targets = targets[np.random.permutation(cls.num_samples)].tolist()  # shuffle

    def test_len(self):
        partitioner = CIFAR10Partitioner(self.targets,
                                         self.num_clients,
                                         balance=True,
                                         partition="iid",
                                         verbose=False,
                                         seed=self.seed)
        self.assertEqual(len(partitioner), self.num_clients)

    def test_item(self):
        partitioner = CIFAR10Partitioner(self.targets,
                                         self.num_clients,
                                         balance=True,
                                         partition="iid",
                                         verbose=False,
                                         seed=self.seed)
        res = [all(partitioner[cid] == partitioner.client_dict[cid]) for cid in
               range(self.num_clients)]
        self.assertTrue(all(res))

    def test_hetero_dir_partition(self):
        # perform partition
        partitioner = CIFAR10Partitioner(self.targets,
                                       self.num_clients,
                                       balance=None,
                                       partition="dirichlet",
                                       dir_alpha=0.3,
                                       verbose=False,
                                       seed=self.seed)
        self.assertEqual(len(partitioner), self.num_clients)
        client_sample_nums = np.array([partitioner[cid].shape[0] for cid in range(self.num_clients)])
        # sample number of each client should no less than number of classes
        self.assertTrue(all(client_sample_nums >= self.num_classes))
        # sample number of each client should not be the same
        self.assertTrue(len(set(client_sample_nums)) >= 2)

    def test_shards_partition(self):
        num_shards = 200
        partitioner = CIFAR10Partitioner(self.targets,
                                       self.num_clients,
                                       balance=None,
                                       partition="shards",
                                       num_shards=num_shards,
                                       verbose=False,
                                       seed=self.seed)
        self.assertEqual(len(partitioner), self.num_clients)
        self.assertTrue(all([partitioner[cid].shape[0] == (
                int(num_shards / self.num_clients) * int(self.num_samples / num_shards)) for cid in
                             range(self.num_clients)]))

    def test_balance_iid_partition(self):
        partitioner = CIFAR10Partitioner(self.targets,
                                       self.num_clients,
                                       balance=True,
                                       partition="iid",
                                       verbose=False,
                                       seed=self.seed)
        self.assertEqual(len(partitioner), self.num_clients)
        # check balance
        self._check_balance(partitioner.client_dict)

    def test_unbalance_iid_partition(self):
        partitioner = CIFAR10Partitioner(self.targets,
                                       self.num_clients,
                                       balance=False,
                                       partition="iid",
                                       unbalance_sgm=0.3,
                                       verbose=False,
                                       seed=self.seed)
        self.assertEqual(len(partitioner), self.num_clients)
        # check unbalance
        self._check_unbalance(partitioner.client_dict)

    def test_balance_dir_partition(self):
        partitioner = CIFAR10Partitioner(self.targets,
                                       self.num_clients,
                                       balance=True,
                                       partition="dirichlet",
                                       dir_alpha=0.3,
                                       verbose=False,
                                       seed=self.seed)
        # check balance
        self._check_balance(partitioner.client_dict)

    def test_unbalance_dir_partition(self):
        partitioner = CIFAR10Partitioner(self.targets,
                                       self.num_clients,
                                       balance=False,
                                       partition="dirichlet",
                                       unbalance_sgm=0.3,
                                       dir_alpha=0.3,
                                       verbose=False,
                                       seed=self.seed)
        # check unbalance
        self._check_unbalance(partitioner.client_dict)

    def _check_unbalance(self, client_dict):
        client_sample_nums = np.array([client_dict[cid].shape[0] for cid in range(self.num_clients)])
        self.assertFalse(all(client_sample_nums == int(self.num_samples / self.num_clients)))

    def _check_balance(self, client_dict):
        client_sample_nums = np.array([client_dict[cid].shape[0] for cid in range(self.num_clients)])
        self.assertTrue(all(client_sample_nums == int(self.num_samples / self.num_clients)))

    def test_invalid_partition(self):
        with self.assertRaises(ValueError):
            partition = CIFAR10Partitioner(self.targets,
                                           self.num_clients,
                                           balance='this',
                                           partition="iid",
                                           verbose=False,
                                           seed=self.seed)


class BasicPartitionerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_classes = 2
        cls.num_clients = 10
        cls.seed = 2021
        targets = []
        for k in range(cls.num_classes):
            targets.extend([k for _ in range(500)])
        cls.num_samples = len(targets)
        targets = np.array(targets)
        np.random.seed(cls.seed)
        cls.targets = targets[np.random.permutation(cls.num_samples)].tolist()  # shuffle

    def test_len(self):
        partitioner = BasicPartitioner(self.targets,
                                         self.num_clients,
                                         partition="iid",
                                         verbose=False,
                                         seed=self.seed)
        self.assertEqual(len(partitioner), self.num_clients)

    def test_item(self):
        partitioner = BasicPartitioner(self.targets,
                                         self.num_clients,
                                         partition="iid",
                                         verbose=False,
                                         seed=self.seed)
        res = [all(partitioner[cid] == partitioner.client_dict[cid]) for cid in
               range(self.num_clients)]
        self.assertTrue(all(res)) 

    def test_noniid_major_classes_partition(self):
        # perform partition='noniid-#label'
        major_classes_num = 1
        partitioner = BasicPartitioner(self.targets,
                                       self.num_clients,
                                       partition="noniid-#label",
                                       major_classes_num=major_classes_num,
                                       verbose=False,
                                       seed=self.seed)
        # basic content check
        self._content_check(partitioner.client_dict)
        # check label class number for each client
        for cid in partitioner.client_dict:
            cid_labels = [self.targets[idx]for idx in partitioner.client_dict[cid]]
            unique_labels = set(cid_labels)
            self.assertEqual(len(unique_labels), major_classes_num)

    def test_noniid_labeldir_partition(self):
        # perform partition='noniid-labeldir'
        partitioner = BasicPartitioner(self.targets,
                                       self.num_clients,
                                       partition="noniid-labeldir",
                                       dir_alpha=0.3,
                                       verbose=False,
                                       seed=self.seed)
        # basic content check
        self._content_check(partitioner.client_dict)
        # sample number for each clients should not be same for all clients
        self._check_unbalance(partitioner.client_dict)

    def test_unbalance_partition(self):
        # perform partition='unbalance'
        partitioner = BasicPartitioner(self.targets,
                                       self.num_clients,
                                       partition="unbalance",
                                       dir_alpha=0.3,
                                       verbose=False,
                                       seed=self.seed)
        # basic content check
        self._content_check(partitioner.client_dict)     
        # sample number for each client should not be equal
        self._check_unbalance(partitioner.client_dict)

    def test_iid_partition(self):
        # perform partition='iid'
        partitioner = BasicPartitioner(self.targets,
                                       self.num_clients,
                                       partition="iid",
                                       verbose=False,
                                       seed=self.seed)
        # basic content check
        self._content_check(partitioner.client_dict)     
        # check balance
        self._check_balance(partitioner.client_dict)

    def _check_unbalance(self, client_dict):
        client_sample_nums = np.array([client_dict[cid].shape[0] for cid in range(self.num_clients)])
        self.assertFalse(all(client_sample_nums == int(self.num_samples / self.num_clients)))

    def _check_balance(self, client_dict):
        client_sample_nums = np.array([client_dict[cid].shape[0] for cid in range(self.num_clients)])
        self.assertTrue(all(client_sample_nums == int(self.num_samples / self.num_clients)))

    def test_invalid_partition(self):
        # perform invalid partition scheme
        with self.assertRaises(ValueError):
            partition = BasicPartitioner(self.targets,
                                         self.num_clients,
                                         partition="this",
                                         verbose=False,
                                         seed=self.seed)

    def _content_check(self, client_dict, num_samples=None):
        # check number of partition parts
        self.assertTrue(len(client_dict) == self.num_clients)
        # total number should equal to num_samples
        all_samples = np.concatenate([client_dict[cid] for cid in range(self.num_clients)]).tolist()
        unique_samples = set(all_samples)
        if num_samples is None:
            num_samples = self.num_samples
        self.assertTrue(unique_samples <= set(list(range(num_samples))))



class VisionPartitionerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_classes = 10
        cls.num_clients = 10
        cls.seed = 2021
        targets = []
        for k in range(cls.num_classes):
            targets.extend([k for _ in range(500)])
        cls.num_samples = len(targets)
        targets = np.array(targets)
        np.random.seed(cls.seed)
        cls.targets = targets[np.random.permutation(cls.num_samples)].tolist()  # shuffle

    def test_init(self):
        verbose = False
        partition = 'iid'
        partitioner = VisionPartitioner(self.targets,
                                        self.num_clients, 
                                        partition=partition,
                                        verbose=verbose, 
                                        seed=self.seed)
        self.assertEqual(self.num_samples, partitioner.num_samples)
        self.assertEqual(self.num_clients, partitioner.num_clients)
        self.assertEqual(partition, partitioner.partition)
        self.assertEqual(verbose, partitioner.verbose)



class FCUBEPartitionerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_classes = 2  # DO NOT change 
        cls.num_clients = 4  # DO NOT change 
        cls.num_samples = 400
        cls.seed = 2021
        # generate fcube data
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(int(cls.num_samples / 4)):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 == 1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)

        cls.data = np.array(X_train, dtype=np.float32)
        cls.targets = np.array(y_train, dtype=np.int32)

    def test_init(self):
        # if data is np.array
        partitioner = FCUBEPartitioner(self.data,
                                       partition="iid")
        self.assertEqual(partitioner.num_samples, self.num_samples)

        # if data is list
        partitioner = FCUBEPartitioner(self.data.tolist(),
                                       partition="iid")
        self.assertEqual(partitioner.num_samples, self.num_samples)

    def test_len(self):
        partitioner = FCUBEPartitioner(self.data,
                                       partition="iid")
        self.assertEqual(len(partitioner), self.num_clients)


    def test_item(self):
        partitioner = FCUBEPartitioner(self.data,
                                       partition="iid")
        res = [all(partitioner[cid] == partitioner.client_dict[cid]) for cid in
               range(self.num_clients)]
        self.assertTrue(all(res)) 

    def test_synthetic_partition(self):
        # perform partition='synthetic'
        partitioner = FCUBEPartitioner(self.data,
                                       partition="synthetic")
        # check client number 
        self.assertEqual(len(partitioner.client_dict), self.num_clients) 
        # check total sample number 
        self.assertEqual(self.num_samples, sum([len(partitioner.client_dict[cid]) for cid in partitioner.client_dict]))  

    def test_iid_partition(self):
        # perform partition='iid'
        partitioner = FCUBEPartitioner(self.data,
                                       partition="iid")
        # check client number 
        self.assertEqual(len(partitioner.client_dict), self.num_clients) 
        # check total sample number 
        self.assertEqual(self.num_samples, sum([len(partitioner.client_dict[cid]) for cid in partitioner.client_dict]))

    def test_invalid_partition(self):
        # perform invalid partition scheme
        with self.assertRaises(ValueError):
            partitioner = FCUBEPartitioner(self.data,
                                        partition="this")