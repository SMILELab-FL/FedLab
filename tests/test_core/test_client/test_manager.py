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

from tkinter.messagebox import Message
import unittest
import time
from copy import deepcopy
from fedlab.core import client

from fedlab.core.client.manager import ClientManager, ActiveClientManager, PassiveClientManager
from fedlab.core.client import ORDINARY_TRAINER, SERIAL_TRAINER
from fedlab.core.network import DistNetwork
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.core.client.trainer import ClientTrainer, SerialClientTrainer
from fedlab.utils import MessageCode, Logger

from ..task_setting_for_test import CNN_Mnist

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

class TestClientTrainer(ClientTrainer):
    def __init__(self, model, cuda=False, device=None, logger=None):
        super().__init__(model, cuda, device)
        self._LOGGER = Logger() if logger is None else logger

    @property
    def uplink_package(self):
        return [torch.tensor([1, 2, 3, 4])]

    def local_process(self, payload, id):
        return None

class TestSerialClientTrainer(SerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False):
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger

    @property
    def uplink_package(self):
        return [torch.tensor([1, 2, 3, 4]) for _ in range(self.num_clients)]

    def local_process(self, payload, id_list):
        return None



class ClientManagerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.host_ip = 'localhost'
        self.model = CNN_Mnist()
        self.port = '3333'

    def test_init(self):
        network = DistNetwork(address=(self.host_ip, self.port),
                              world_size=1,
                              rank=0)
        trainer = ClientTrainer(model=self.model, cuda=False) 
        client_manager = ClientManager(network, trainer)
        self.assertIsInstance(client_manager._network, DistNetwork)
        self.assertIsInstance(client_manager._trainer, ModelMaintainer)
    
    def test_setup(self):
        server_rank = 0
        client_rank = 1

        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=server_rank)
        client_network = DistNetwork(address=(self.host_ip, self.port),
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


class PassiveClientManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.host_ip = 'localhost'
        self.port = '3333'
        self.server_rank = 0
        self.client_rank = 1
        self.model = CNN_Mnist()

    def test_init(self):
        self._check_init_ordinary_trainer()
        self._check_init_serial_trainer()

    def test_synchronize(self):
        self._check_synchronize_ordinary_trainer()
        self._check_synchronize_serial_trainer()

    def test_main_loop(self):
        self._check_main_loop_ordinary_trainer()
        self._check_main_loop_serial_trainer()

    def _check_init_serial_trainer(self):
        trainer = ClientTrainer(model=self.model, cuda=False)
        network = DistNetwork(address=(self.host_ip, self.port),
                              world_size=1,
                              rank=0)
        manager = PassiveClientManager(network=network, trainer=trainer)
        self.assertIsInstance(manager._network, DistNetwork)
        self.assertIsInstance(manager._trainer, ModelMaintainer)

    def _check_init_ordinary_trainer(self):
        trainer = SerialClientTrainer(model=self.model, num_clients=10, cuda=False)
        network = DistNetwork(address=(self.host_ip, self.port),
                              world_size=1,
                              rank=0)
        manager = PassiveClientManager(network=network, trainer=trainer)
        self.assertIsInstance(manager._network, DistNetwork)
        self.assertIsInstance(manager._trainer, ModelMaintainer)
    
    def _check_synchronize_ordinary_trainer(self):
        num_clients = 1
        check_content = [torch.tensor([1,2,3,4])]

        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=0)
        client_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=1)
        trainer = TestClientTrainer(model=self.model, cuda=False) 
        client_manager = PassiveClientManager(client_network, trainer)
        
        server = Process(target=self._run_server_synchronize, 
                         args=(server_network, 1, num_clients, check_content))
        client = Process(target=self._run_client_synchronize, args=(client_manager,))

        server.start()
        client.start()

        server.join()
        client.join()

    def _check_main_loop_ordinary_trainer(self):
        num_clients = 1

        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=0)
        client_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=1)
        trainer = TestClientTrainer(model=self.model, cuda=False) 
        client_manager = PassiveClientManager(client_network, trainer)
        
        server = Process(target=self._run_server_main_loop, 
                         args=(server_network, 1, num_clients))
        client = Process(target=self._run_client_main_loop, args=(client_manager,))

        server.start()
        client.start()

        server.join()
        client.join()

    def _check_main_loop_serial_trainer(self):
        num_clients = 5

        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=0)
        client_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=1)
        trainer = TestSerialClientTrainer(model=self.model, cuda=False, num_clients=num_clients) 
        client_manager = PassiveClientManager(client_network, trainer)
        
        server = Process(target=self._run_server_main_loop, 
                         args=(server_network, 1, num_clients))
        client = Process(target=self._run_client_main_loop, args=(client_manager,))

        server.start()
        client.start()

        server.join()
        client.join()

    def _run_client_main_loop(self, client_manager):
        client_manager._network.init_network_connection()
        client_manager.main_loop()
        client_manager._network.close_network_connection()

    def _run_server_main_loop(self, server_network, client_rank, num_clients):
        server_network.init_network_connection()
        # Parameter Update
        server_network.send(content=[torch.tensor([0]), torch.tensor([1,2,3,4])], 
                            message_code=MessageCode.ParameterUpdate, 
                            dst=client_rank)
        # synchronize stage
        for _ in range(num_clients):
            _, message_code, _ = server_network.recv(src=client_rank)
            self.assertEqual(message_code, MessageCode.ParameterUpdate)

        # Exit
        server_network.send(message_code=MessageCode.Exit, dst=client_rank)
        _, message_code, _ = server_network.recv(src=client_rank)
        self.assertEqual(message_code, MessageCode.Exit)

        server_network.close_network_connection()


    def _run_server_synchronize(self, server_network, client_rank, num_clients, check_content):
        server_network.init_network_connection()
        for _ in range(num_clients):
            _, message_code, content = server_network.recv(src=client_rank)
            self._check_content_eq(content, check_content)
            self.assertEqual(message_code, MessageCode.ParameterUpdate)
        server_network.close_network_connection()

    def _run_client_synchronize(self, client_manager):
        client_manager._network.init_network_connection()
        client_manager.synchronize()
        client_manager._network.close_network_connection()

    def _check_synchronize_serial_trainer(self):
        num_clients = 5
        check_content = [torch.tensor([1,2,3,4])]

        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=0)
        client_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=1)
        trainer = TestSerialClientTrainer(model=self.model, num_clients=num_clients, cuda=False) 
        client_manager = PassiveClientManager(client_network, trainer)
        
        server = Process(target=self._run_server_synchronize, 
                         args=(server_network, self.client_rank, num_clients, check_content))
        client = Process(target=self._run_client_synchronize, args=(client_manager,))

        server.start()
        client.start()

        server.join()
        client.join()

    def _check_content_eq(self, content, check_content):
        check_res = [torch.equal(t, p_t) for t, p_t in zip(content, check_content)]
        return all(check_res)


class ActiveClientManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.host_ip = 'localhost'
        self.port = '5555'
        self.server_rank = 0
        self.client_rank = 1
        self.model = CNN_Mnist()

    def test_init(self):
        trainer = ClientTrainer(model=self.model, cuda=False)
        network = DistNetwork(address=(self.host_ip, self.port),
                              world_size=1,
                              rank=0)
        manager = ActiveClientManager(network=network, trainer=trainer)
        self.assertIsInstance(manager._network, DistNetwork)
        self.assertIsInstance(manager._trainer, ModelMaintainer)
    
    def test_synchronize(self):
        check_content = [torch.tensor([1,2,3,4])]

        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=0)
        client_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=1)
        trainer = TestClientTrainer(model=self.model, cuda=False) 
        client_manager = ActiveClientManager(client_network, trainer)
        
        server = Process(target=self._run_server_synchronize, 
                         args=(server_network, 1, check_content))
        client = Process(target=self._run_client_synchronize, args=(client_manager,))

        server.start()
        client.start()

        server.join()
        client.join()

    def test_request(self):
        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=0)
        client_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=1)
        trainer = TestClientTrainer(model=self.model, cuda=False) 
        client_manager = ActiveClientManager(client_network, trainer)
        
        server = Process(target=self._run_server_request, 
                         args=(server_network, 1))
        client = Process(target=self._run_client_request, args=(client_manager,))

        server.start()
        client.start()

        server.join()
        client.join()

    def test_main_loop(self):
        self._check_main_loop_oridinary_trainer()
        self._check_main_loop_serial_trainer()

    def _check_main_loop_oridinary_trainer(self):
        check_content = [torch.tensor([1,2,3,4])]

        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=self.server_rank)
        client_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=self.client_rank)
        trainer = TestClientTrainer(model=self.model, cuda=False) 
        client_manager = ActiveClientManager(client_network, trainer)
        
        server = Process(target=self._run_server_main_loop, 
                         args=(server_network, 1, check_content))
        client = Process(target=self._run_client_main_loop, args=(client_manager,))

        server.start()
        client.start()

        server.join()
        client.join()

    def _check_main_loop_serial_trainer(self):
        check_content = [torch.tensor([1,2,3,4])]
        num_clients = 5

        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=self.server_rank)
        client_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=2,
                                     rank=self.client_rank)
        trainer = TestSerialClientTrainer(model=self.model, cuda=False, num_clients=num_clients) 
        client_manager = ActiveClientManager(client_network, trainer)
        
        server = Process(target=self._run_server_main_loop, 
                         args=(server_network, 1, check_content))
        client = Process(target=self._run_client_main_loop, args=(client_manager,))

        server.start()
        client.start()

        server.join()
        client.join()

    def _run_client_main_loop(self, client_manager):
        client_manager._network.init_network_connection()
        client_manager.main_loop()
        client_manager._network.close_network_connection()

    def _run_client_synchronize(self, client_manager):
        client_manager._network.init_network_connection()
        client_manager.synchronize()
        client_manager._network.close_network_connection()

    def _run_client_request(self, client_manager):
        client_manager._network.init_network_connection()
        client_manager.request()
        client_manager._network.close_network_connection()

    def _run_server_main_loop(self, server_network, client_rank, check_content):
        server_network.init_network_connection()
        # recv request from client
        _, message_code, _ = server_network.recv(src=client_rank)
        self.assertEqual(message_code, MessageCode.ParameterRequest)
        # send ParameterUpdate signal
        server_network.send(content=[torch.tensor([0]), torch.tensor([1,2,3,4])], 
                            message_code=MessageCode.ParameterUpdate, 
                            dst=client_rank)
        # synchronize stage
        _, message_code, _ = server_network.recv(src=client_rank)
        self.assertEqual(message_code, MessageCode.ParameterUpdate)
        # recv request from client
        _, message_code, _ = server_network.recv(src=client_rank)
        self.assertEqual(message_code, MessageCode.ParameterRequest)
        # Exit stage
        server_network.send(message_code=MessageCode.Exit, dst=client_rank)
        _, message_code, _ = server_network.recv(src=client_rank)
        self.assertEqual(message_code, MessageCode.Exit)
        server_network.close_network_connection()

    def _run_server_synchronize(self, server_network, client_rank, check_content):
        server_network.init_network_connection()
        _, message_code, content = server_network.recv(src=client_rank)
        self._check_content_eq(content, check_content)
        self.assertEqual(message_code, MessageCode.ParameterUpdate)
        server_network.close_network_connection()

    def _run_server_request(self, server_network, client_rank):
        server_network.init_network_connection()
        _, message_code, _ = server_network.recv(src=client_rank)
        self.assertEqual(message_code, MessageCode.ParameterRequest)
        server_network.close_network_connection()
    

    def _check_content_eq(self, content, check_content):
        check_res = [torch.equal(t, p_t) for t, p_t in zip(content, check_content)]
        return all(check_res)