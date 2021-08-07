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
import torch
from fedlab_benchmarks.datasets.leaf_data_process.data_read_util import read_dir
from fedlab_benchmarks.datasets.leaf_data_process.dataset.femnist_dataset import FemnistDataset


class LeafTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.datapath = '../../../tests/data/leaf'
        self.dataset = 'femnist'
        self.client_id_name_map = {4: 'f3797_07', 3: 'f3793_06', 1: 'f3728_28', 0: 'f3687_48', 2: 'f3785_26'}
        self.train_sample_num_map = {'f3797_07': 11, 'f3793_06': 16, 'f3728_28': 17, 'f3687_48': 18, 'f3785_26': 16}
        self.test_sample_num_map = {'f3797_07': 19, 'f3793_06': 17, 'f3728_28': 19, 'f3687_48': 18, 'f3785_26': 16}

    def tearDown(self) -> None:
        return super().tearDown()

    def test_get_data_client(self):
        train_path = os.path.join(self.datapath, 'train')
        client_id2name_train, _, client_name2data_train = read_dir(train_path)
        test_path = os.path.join(self.datapath, 'test')
        client_id2name_test, _, client_name2data_test = read_dir(test_path)
        self.assertEqual(self.client_id_name_map, client_id2name_train, client_id2name_test)

        for client_id in range(5):
            client_name = client_id2name_train(client_id)
            sample_num = len(client_id2name_train(client_name)['x'])
            assert sample_num == self.train_sample_num_map[client_name]
            client_name = client_id2name_test(client_id)
            sample_num = len(client_id2name_test(client_name)['x'])
            assert sample_num == self.test_sample_num_map[client_name]

    def test_data_process_leaf(self):
        for client_id in range(5):
            trainset = FemnistDataset(client_id=client_id, data_root=self.datapath, is_train=True)
            assert trainset.data.shape[1:] == (1, 28, 28)
            assert isinstance(trainset.targets[0], torch.LongTensor)
