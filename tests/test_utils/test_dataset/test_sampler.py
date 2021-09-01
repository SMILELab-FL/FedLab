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
from fedlab.utils.dataset.sampler import SubsetSampler, RawPartitionSampler, DictFileSampler


class SamplerTestCase(unittest.TestCase):
    def test_sampler(self):
        indices = [i for i in range(1000)]
        sampler = SubsetSampler(indices=indices)

        for x, y in zip(sampler, indices):
            assert x == y

        assert len(indices) == len(sampler)

    def test_raw_partition_sampler(self):

        indices = [i for i in range(1000)]
        sampler = RawPartitionSampler(indices, num_replicas=10, client_id=1)

        for _ in sampler:
            break

        assert len(sampler) == len(indices) / 10

    def test_dict_partition_sampler(self):
        dict_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 "../../data/mnist_iid.pkl")
        sampler = DictFileSampler(dict_file=dict_file, client_id=10)
        for _ in sampler:
            break

        len(sampler)
