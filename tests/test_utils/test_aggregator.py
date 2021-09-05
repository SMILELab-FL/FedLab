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
import random
import torch
import numpy as np
from fedlab.utils.aggregator import Aggregators


class AggregatorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shape = (10000,)
        cls.params = []
        cls.num_params = random.randint(1, 10)
        for _ in range(cls.num_params):
            cls.params.append(torch.rand(size=cls.shape))

    def test_fedavg_aggregate_no_weight(self):
        merged_params = Aggregators.fedavg_aggregate(self.params)
        no_weight_res = torch.zeros(size=self.shape)
        for param in self.params:
            no_weight_res += param
        no_weight_res /= self.num_params

        no_weight_diff = torch.max(torch.abs(no_weight_res - merged_params)).numpy()

        self.assertAlmostEqual(0, no_weight_diff, 6)

    def test_fedavg_aggregate_weighted(self):
        list_weights = [random.random() for _ in range(self.num_params)]
        np_weights = np.array(list_weights)
        torch_weights = torch.tensor(list_weights)

        normed_weight = [weight / sum(list_weights) for weight in list_weights]
        weight_res = torch.zeros(size=self.shape)
        for i, param in enumerate(self.params):
            weight_res += normed_weight[i] * param

        np_res = Aggregators.fedavg_aggregate(self.params, np_weights)
        list_res = Aggregators.fedavg_aggregate(self.params, list_weights)
        torch_res = Aggregators.fedavg_aggregate(self.params, torch_weights)

        np_diff = torch.max(torch.abs(np_res - weight_res)).numpy()
        list_diff = torch.max(torch.abs(list_res - weight_res)).numpy()
        torch_diff = torch.max(torch.abs(torch_res - weight_res)).numpy()

        self.assertAlmostEqual(0, np_diff, 6, 'numpy.array weights not pass')
        self.assertAlmostEqual(0, list_diff, 6, 'list weights not pass')
        self.assertAlmostEqual(0, torch_diff, 6, 'torch.Tensor weights not pass')

    def test_fedasgd_aggregate(self):

        server_params = torch.rand(size=self.shape)
        comming_params = torch.rand(size=self.shape)

        merged_params = Aggregators.fedasync_aggregate(server_params, comming_params, 0.5)

        assert self.shape == merged_params.shape
