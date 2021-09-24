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

sys.path.append("../")
from copy import deepcopy

from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.core.client.manager import ClientPassiveManager
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.network import DistNetwork

from tests.test_core.task_setting_for_test import (
    unittest_dataloader,
    model,
    criterion,
    optimizer,
)

class FedAvgClientTestCase(unittest.TestCase):
    def setUp(self) -> None:
        ip = "127.0.0.1"
        port = "3003"
        world_size = 2

        ps = SyncParameterServerHandler(deepcopy(model))
        self.server = ServerSynchronousManager(
            handler=ps,
            network=DistNetwork(address=(ip, port),
                                world_size=world_size,
                                rank=0),
        )

        self.server.start()

        dataloader = unittest_dataloader()
        trainer = ClientSGDTrainer(
            model,
            dataloader,
            epochs=1,
            optimizer=optimizer,
            criterion=criterion,
            cuda=False,
        )
        self.client = ClientPassiveManager(trainer=trainer,
                                           network=DistNetwork(
                                               address=(ip, port),
                                               world_size=world_size,
                                               rank=1))

    def tearDown(self) -> None:
        pass

    def test_fedavg(self):
        self.client.run()
