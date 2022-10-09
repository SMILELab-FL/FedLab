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
import time

from fedlab.core.client.manager import ClientManager, ActiveClientManager, PassiveClientManager
from fedlab.core.network import DistNetwork
from fedlab.core.client.trainer import ClientTrainer
from fedlab.utils.message_code import MessageCode

from ..task_setting_for_test import CNN_Mnist

import torch
import torch.distributed as dist
from torch.multiprocessing import Process


class ClientManagerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.host_ip = 'localhost'
        self.model = CNN_Mnist()

    def test_init(self):
        port = '3333'
        network = DistNetwork(address=(self.host_ip, port),
                              world_size=1,
                              rank=0)
        trainer = ClientTrainer(model=self.model, cuda=False) 
        client_manager = ClientManager(network, trainer)
        self.assertIsInstance(client_manager._network, DistNetwork)
        self.assertIsInstance(client_manager._trainer, ClientTrainer)
    
    def test_setup(self):
        server_rank = 0
        client_rank = 1
        port = '3333'

        server_network = DistNetwork(address=(self.host_ip, port),
                                     world_size=2,
                                     rank=server_rank)
        client_network = DistNetwork(address=(self.host_ip, port),
                                     world_size=2,
                                     rank=client_rank)
        trainer = ClientTrainer(model=self.model, cuda=False) 
        client_manager = ClientManager(client_network, trainer)
        
        server = Process(target=self._run_server, args=(server_network, client_rank))
        client = Process(target=self._run_client_manager, args=(client_manager,))

        server.start()
        client.start()

        server.join()
        client.join()

    def _run_server(self, server_network, src):
        server_network.init_network_connection()
        client_rank, message_code, content = server_network.recv(src=src)
        self.assertEqual(client_rank, src)
        self.assertEqual(message_code, MessageCode.SetUp)
        server_network.close_network_connection()

    def _run_client_manager(self, client_manager):
        client_manager.setup()
        client_manager.shutdown()

