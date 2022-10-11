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
from copy import deepcopy

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
        return [self.model_parameters]

    def local_process(self, payload, id):
        pass 

class TestSerialClientTrainer(SerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False):
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger
        self.cache = []

    @property
    def uplink_package(self):
        self.cache = [[self.model_parameters] for _ in range(self.num_clients)]
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list):
        pass


class TestPassiveClientManager(PassiveClientManager):
    def __init__(self, network, trainer, logger=None):
        super().__init__(network, trainer, logger)

    def synchronize(self):
        """Synchronize with server."""
        self._LOGGER.info("Uploading information to server.")
        if self._trainer.type == SERIAL_TRAINER:
            payloads = self._trainer.uplink_package
            for idx, elem in enumerate(payloads):
                self._LOGGER.info("SERIAL_TRAINER trying to synchronize sending client-{idx}'s information...")
                self._network.send(content=elem,
                                message_code=MessageCode.ParameterUpdate,
                                dst=0)
                self._LOGGER.info("SERIAL_TRAINER synchronize client-{idx} done.")
                            
        if self._trainer.type == ORDINARY_TRAINER:
            self._LOGGER.info("ORDINARY_TRAINER trying to synchronize sending...")
            self._network.send(content=self._trainer.uplink_package,
                                message_code=MessageCode.ParameterUpdate,
                                dst=0)
            self._LOGGER.info("ORDINARY_TRAINER synchronize done.")



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

    # def test_synchronize(self):
    #     self._check_synchronize_ordinary_trainer()
    #     # self._check_synchronize_serial_trainer()

    # def test_main_loop(self):
    #     # self._check_main_loop_ordinary_trainer()
    #     # self._check_main_loop_serial_trainer()
    #     pass

    def _check_init_serial_trainer(self):
        trainer = ClientTrainer(model=self.model, cuda=False)
        network = DistNetwork(address=(self.host_ip, self.port),
                              world_size=1,
                              rank=0)
        manager = ClientManager(network=network, trainer=trainer)
        self.assertIsInstance(manager._network, DistNetwork)
        self.assertIsInstance(manager._trainer, ModelMaintainer)

    def _check_init_ordinary_trainer(self):
        trainer = SerialClientTrainer(model=self.model, num_clients=10, cuda=False)
        network = DistNetwork(address=(self.host_ip, self.port),
                              world_size=1,
                              rank=0)
        manager = ClientManager(network=network, trainer=trainer)
        self.assertIsInstance(manager._network, DistNetwork)
        self.assertIsInstance(manager._trainer, ModelMaintainer)

    
    # def _check_synchronize_ordinary_trainer(self):
    #     num_clients = 1

    #     server_network = DistNetwork(address=(self.host_ip, self.port),
    #                                  world_size=2,
    #                                  rank=self.server_rank)
    #     client_network = DistNetwork(address=(self.host_ip, self.port),
    #                                  world_size=2,
    #                                  rank=self.client_rank)
    #     trainer = TestClientTrainer(model=self.model, cuda=False) 
    #     client_manager = TestPassiveClientManager(client_network, trainer)  # TODO: use test class 

    #     check_contents = [[deepcopy(trainer.model_parameters)] for _ in range(num_clients)]
        
    #     server = Process(target=self._run_server_synchronize, 
    #                      args=(server_network, self.client_rank, num_clients, check_contents))
    #     client = Process(target=self._run_client_synchronize, args=(client_manager,))

    #     server.start()
    #     client.start()

    #     server.join()
    #     client.join()

    # def _check_synchronize_serial_trainer(self):
    #     num_clients = 5

    #     server_network = DistNetwork(address=(self.host_ip, self.port),
    #                                  world_size=2,
    #                                  rank=self.server_rank)
    #     client_network = DistNetwork(address=(self.host_ip, self.port),
    #                                  world_size=2,
    #                                  rank=self.client_rank)
    #     trainer = TestSerialClientTrainer(model=self.model, num_clients=num_clients, cuda=False) 
    #     client_manager = PassiveClientManager(client_network, trainer)

    #     check_contents = [[deepcopy(trainer.model_parameters)] for _ in range(num_clients)]
        
    #     server = Process(target=self._run_server_synchronize, 
    #                      args=(server_network, self.client_rank, num_clients, check_contents))
    #     client = Process(target=self._run_client_synchronize, args=(client_manager,))

    #     server.start()
    #     client.start()

    #     server.join()
    #     client.join()

    # def _run_server_synchronize(self, server_network, src, num_clients, check_contents):
    #     # init sercer netnetwork
    #     server_network.init_network_connection()
    #     print(f"Server init done")
    #     # # receive setup package
    #     _, message_code, contents = server_network.recv(src=src)
    #     print(f"Server received from client-{num_clients-1}")
    #     # self.assertEqual(message_code, MessageCode.SetUp)
    #     # receive synchronization package
    #     # for i in range(num_clients):
    #     #     print(f"Server waiting for client-{i}'s package")
    #     #     _, message_code, content = server_network.recv(src=src)
    #     #     print(f"Server received from client-{i}")
    #     #     self.assertEqual(message_code, MessageCode.ParameterUpdate)
    #     #     self._check_content_eq(content, check_contents[i])
    #     # shutdown server network
    #     server_network.close_network_connection()

    # def _run_client_synchronize(self, client_manager):
    #     # 1. setup stage: init network manually
    #     client_manager._network.init_network_connection()
    #     # 2. synchronize
    #     print(f"Client Start to synchronize")
    #     client_manager.synchronize()
    #     print(f"Client synchronize done")
    #     # 3. shutdown client network manually
    #     client_manager._network.close_network_connection()
   

    # def _check_main_loop_serial_trainer(self):
    #     server_rank = 0
    #     client_rank = 1

    #     server_network = DistNetwork(address=(self.host_ip, self.port),
    #                                  world_size=2,
    #                                  rank=server_rank)
    #     client_network = DistNetwork(address=(self.host_ip, self.port),
    #                                  world_size=2,
    #                                  rank=client_rank)
    #     trainer = SerialClientTrainer(model=self.model, num_clients=10, cuda=False) 
    #     client_manager = PassiveClientManager(client_network, trainer)
        
    #     server = Process(target=self._run_server, args=(server_network, client_rank))  # TODO
    #     client = Process(target=self._run_client_manager, args=(client_manager,))  # TODO

    #     server.start()
    #     client.start()

    #     server.join()
    #     client.join()

    # def _run_client_main_loop(self, client_manager):
    #     # 1. setup stage
    #     client_manager._network.init_network_connection()

    #     # TODO

    # def _run_server_main_loop(self, server_network, src, num_clients):
    #     server_network.init_network_connection()
    #     # 1. setup stage
    #     client_rank, message_code, content = server_network.recv(src=src)
    #     self.assertEqual(client_rank, src)
    #     self.assertEqual(message_code, MessageCode.SetUp)
    #     self.assertEqual(content, num_clients)  # TODO: maybe torch.tensor?
    #     #TODO

    # def _check_main_loop_ordinary_trainer(self):
    #     server_rank = 0
    #     client_rank = 1

    #     server_network = DistNetwork(address=(self.host_ip, self.port),
    #                                  world_size=2,
    #                                  rank=server_rank)
    #     client_network = DistNetwork(address=(self.host_ip, self.port),
    #                                  world_size=2,
    #                                  rank=client_rank)
    #     trainer = ClientTrainer(model=self.model, cuda=False) 
    #     client_manager = PassiveClientManager(client_network, trainer)
        
    #     server = Process(target=self._run_server, args=(server_network, client_rank))  # TODO
    #     client = Process(target=self._run_client_manager, args=(client_manager,))  # TODO

    #     server.start()
    #     client.start()

    #     server.join()
    #     client.join()

    def _check_content_eq(self, content, check_content):
        check_res = [torch.equal(t, p_t) for t, p_t in zip(content, check_content)]
        return all(check_res)


# class ActiveClientManagerTestCase(unittest.TestCase):
#     def setUp(self) -> None:
#         self.host_ip = 'localhost'
#         self.model = CNN_Mnist()
#         self.port = '3333'

#     def test_init(self):
#         self._check_init_ordinary_trainer()
#         self._check_init_serial_trainer()

#     def _check_init_serial_trainer(self):
#         # TODO: check network, check trainer, check logger
#         pass 

#     def _check_init_ordinary_trainer(self):
#         # TODO: check network, check trainer, check logger
#         pass 

#     def test_request(self):
#         # TODO: use multiprocessing to perform interconnection between server and client
#         pass 

#     def test_synchronize(self):
#         # TODO: use multiprocessing to perform interconnection between server and client
#         pass
