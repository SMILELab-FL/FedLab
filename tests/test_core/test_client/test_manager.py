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

import sys
sys.path.append("../../../")
from fedlab.core.client.manager import ClientActiveManager, ClientPassiveManager
from fedlab.core.client.trainer import ClientTrainer
from fedlab.core.network import DistNetwork


class TestTrainer(ClientTrainer):
    def __init__(self):
        # super().__init__(model, cuda)
        pass

    def train(self):
        pass


class ClientManagerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_init(self):
        trainer = TestTrainer()
        network = DistNetwork(
            address=("127.0.0.1", "3002"), world_size=1, rank=0, ethernet=None
        )

        cam = ClientActiveManager(trainer=trainer, network=network)

        cpm = ClientPassiveManager(trainer=trainer, network=network)
