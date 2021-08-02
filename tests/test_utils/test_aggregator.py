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
from fedlab_utils.aggregator import Aggregators


class AggregatorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shape = (10000,)

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_fedavg_aggregate(self):
        params = []
        for _ in range(random.randint(1, 10)):
            params.append(torch.Tensor(size=self.shape))

        merged_params = Aggregators.fedavg_aggregate(params)

        assert self.shape == merged_params.shape

    def test_fedasgd_aggregate(self):
        
        server_params = torch.Tensor(size=self.shape)
        comming_params = torch.Tensor(size=self.shape)

        merged_params = Aggregators.fedasgd_aggregate(server_params, comming_params, 0.5)

        assert self.shape == merged_params.shape

