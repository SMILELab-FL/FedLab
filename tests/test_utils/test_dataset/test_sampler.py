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

from fedlab.utils.dataset.sampler import SubsetSampler, FedDistributedSampler

class SamplerTestCase(unittest.TestCase):

    def test_sampler(self):
        indices = [i for i in range(1000)]
        samer = SubsetSampler(indices=indices, shuffle=True)

        for idx in samer:
            break

        assert len(indices) == len(samer)

    def test_fed_sampler(self):
        indices = [i for i in range(1000)]
        fed_samer = FedDistributedSampler(indices, num_replicas=10, client_id=1)

        for idx in fed_samer:
            break

        assert len(fed_samer) == len(indices)/10

