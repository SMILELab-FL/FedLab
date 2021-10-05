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
from fedlab.utils.dataset import functional as F


class DatasetFunctionalTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_samples = 1000
        cls.num_classes = 5
        cls.num_clients = 20

    def setUp(self) -> None:
        np.random.seed(2021)

    def test_split_indices(self):
        client_sample_nums = np.ones(self.num_clients) * int(self.num_samples / self.num_clients)
        num_cumsum = np.cumsum(client_sample_nums).astype(int)
        client_dict = F.split_indices(num_cumsum, np.arange(self.num_samples))
        self.assertEqual(len(client_dict), self.num_clients)
        self.assertTrue(self._dict_item_len_check(client_dict, client_sample_nums))
        self.assertTrue(self._dict_element_check(client_dict, np.arange(self.num_samples)))

    def _dict_item_len_check(self, client_dict, client_sample_nums):
        for cid in range(self.num_clients):
            if len(client_dict[cid]) != client_sample_nums[cid]:
                return False
        return True

    def _dict_element_check(self, client_dict, indices):
        concat_dict = np.concatenate([client_dict[cid] for cid in range(self.num_clients)])
        all_equal = all(np.array(indices) == concat_dict)
        return all_equal

    def test_balance_partition(self):
        client_sample_nums = F.balance_split(self.num_clients, self.num_samples)
        self.assertTrue(client_sample_nums.shape[0] == self.num_clients)
        self.assertTrue(all(client_sample_nums == int(self.num_samples / self.num_clients)))

    def test_lognormal_unbalance_split(self):
        # check length
        clt_sample_nums = F.lognormal_unbalance_split(self.num_clients,
                                                      self.num_samples, 0.6)
        self.assertTrue(clt_sample_nums.shape[0] == self.num_clients)
        # sample number for each client should not be equal
        self.assertFalse(all(clt_sample_nums == int(self.num_samples / self.num_clients)))

        # when unbalanced_sgm=0, should have same result as balanced
        lognormal_clt_sample_nums = F.lognormal_unbalance_split(self.num_clients,
                                                                self.num_samples, 0)
        balance_clt_sample_nums = F.balance_split(self.num_clients, self.num_samples)
        self.assertTrue(all(lognormal_clt_sample_nums == balance_clt_sample_nums))

    def test_hetero_dir_partition(self):
        # use np.ndarray targets
        targets = np.random.randint(self.num_classes, size=self.num_samples)
        client_dict = F.hetero_dir_partition(targets,
                                             self.num_clients,
                                             self.num_classes, 0.3)
        self._hetero_content_check(client_dict)

        # use list targets
        client_dict = F.hetero_dir_partition(targets.tolist(),
                                             self.num_clients,
                                             self.num_classes, 0.3)
        self._hetero_content_check(client_dict)

    def _hetero_content_check(self, client_dict):
        # check client number
        self.assertTrue(len(client_dict) == self.num_clients)
        # sample number for each clients should not be same for all clients
        self.assertFalse(
            all([len(client_dict[cid]) == int(self.num_samples / self.num_clients) for cid in
                 range(self.num_clients)]))
        # total number should equal to num_samples
        all_samples = np.concatenate([client_dict[cid] for cid in range(self.num_clients)]).tolist()
        unique_samples = set(all_samples)
        self.assertTrue(unique_samples == set(list(range(self.num_samples))))

    def test_shards_partition(self):
        num_shards = 200
        targets = np.random.randint(self.num_classes, size=self.num_samples)

        # use np.ndarray targets
        client_dict = F.shards_partition(targets, self.num_clients, num_shards)
        self._shards_content_check(client_dict, num_shards)

        # use list targets
        client_dict = F.shards_partition(targets.tolist(), self.num_clients, num_shards)
        self._shards_content_check(client_dict, num_shards)

        # check warnings when num_shards is not dividable for num_clients and num_samples
        with self.assertWarns(Warning,
                              msg="warning: length of dataset isn't divided exactly by num_shards. ") as cm:
            F.shards_partition(targets, self.num_clients, 299)

    def _shards_content_check(self, client_dict, num_shards):
        # check number of partition parts
        self.assertTrue(len(client_dict) == self.num_clients)
        # number of samples should be equal for all clients
        num_samples_per_clients = int(self.num_samples / num_shards) * int(
            num_shards / self.num_clients)
        self.assertTrue(all([len(client_dict[cid]) == num_samples_per_clients for cid in
                             range(self.num_clients)]))

    def test_client_inner_dirichlet_partition(self):
        targets = np.random.randint(self.num_classes, size=self.num_samples)
        client_sample_nums = F.balance_split(self.num_clients, self.num_samples)

        # use np.ndarray targets
        client_dict = F.client_inner_dirichlet_partition(targets,
                                                         self.num_clients,
                                                         self.num_classes,
                                                         dir_alpha=0.3,
                                                         client_sample_nums=client_sample_nums.copy(),
                                                         verbose=False)
        self._client_inner_dirichlet_content_check(client_dict, client_sample_nums)

        # use list targets
        client_dict = F.client_inner_dirichlet_partition(targets.tolist(),
                                                         self.num_clients,
                                                         self.num_classes,
                                                         dir_alpha=0.3,
                                                         client_sample_nums=client_sample_nums.copy(),
                                                         verbose=False)
        self._client_inner_dirichlet_content_check(client_dict, client_sample_nums)

        # check verbose
        F.client_inner_dirichlet_partition([0, 0, 1, 1, 2, 1, 3, 3, 4, 4],
                                           num_clients=2,
                                           num_classes=5,
                                           dir_alpha=0.3,
                                           client_sample_nums=np.array([5, 5]).astype(int),
                                           verbose=True)

    def _client_inner_dirichlet_content_check(self, client_dict, client_sample_nums):
        # check number of partition parts
        self.assertTrue(len(client_dict) == self.num_clients)
        # check sample number for each client
        self.assertTrue(all([client_dict[cid].shape[0] == client_sample_nums[cid] for cid in
                             range(self.num_clients)]))
