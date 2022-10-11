import unittest
import time
from random import randint
from copy import deepcopy

from fedlab.core.client import ORDINARY_TRAINER, SERIAL_TRAINER
from fedlab.core.network import DistNetwork
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.core.server.handler import ServerHandler
from fedlab.core.server.manager import ServerManager, SynchronousServerManager, AsynchronousServerManager
from fedlab.utils import MessageCode, Logger, message_code

from ..task_setting_for_test import CNN_Mnist

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

class TestServerHandler(ServerHandler):
    def __init__(self, model, cuda=False, device=None):
        super().__init__(model, cuda, device)

class ServerManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.host_ip = 'localhost'
        self.model = CNN_Mnist()

    def test_init(self):
        self._check_init(mode="LOCAL")
        self._check_init(mode="GLOBAL")

    def _check_init(self, mode):
        port = '3444'
        network = DistNetwork(address=(self.host_ip, port),
                              world_size=1,
                              rank=0)
        handler = TestServerHandler(self.model, cuda=False)
        manager = ServerManager(network, handler, mode=mode)
        self.assertIsInstance(manager._handler, ServerHandler)
        self.assertEqual(manager.mode, mode)
        self.assertEqual(manager.coordinator, None)

    
    def test_setup(self):
        port = '5555'
        client_rank_num = 4
        num_clients_list = [randint(3,10) for _ in range(client_rank_num)]  # number of clients for each client trainer
        mode = "LOCAL"

        # set server network
        server_network = DistNetwork(address=(self.host_ip, port),
                                     world_size=1 + client_rank_num,
                                     rank=0)
        handler = TestServerHandler(self.model, cuda=False)
        server_manager = ServerManager(server_network, handler, mode=mode)
        server_p = Process(target=self._run_server_manager_setup,
                           args=(server_manager, num_clients_list))
        

        client_networks = []
        client_proceses = []
        for rank in range(1, client_rank_num + 1):
            client_network = DistNetwork(address=(self.host_ip, port),
                                         world_size=1 + client_rank_num,
                                         rank=rank)
            client_networks.append(client_network)
            client_p = Process(target=self._run_client_network_setup,
                               args=(client_network, num_clients_list[rank-1], rank))
            client_proceses.append(client_p)
        
        server_p.start()
        for client_p in client_proceses:
            client_p.start()

        server_p.join()
        for client_p in client_proceses:
            client_p.join()
    

    def _run_client_network_setup(self, client_network, num_clients, rank):
        client_network.init_network_connection()
        # time.sleep(5)
        client_network.send(content=torch.tensor(num_clients).int(),
                            message_code=MessageCode.SetUp,
                            dst=0)
        client_network.close_network_connection()

    def _run_server_manager_setup(self, server_manager, num_clients_list):
        server_manager.setup()
        self._check_coordinator_map(server_manager.coordinator.map, num_clients_list)
        self.assertEqual(server_manager._handler.num_clients, sum(num_clients_list))
        server_manager.shutdown()

    def _check_coordinator_map(self, coor_map, num_clients_list):
        self.assertEqual(len(num_clients_list), len(coor_map))
        for rank in coor_map:
            self.assertEqual(coor_map[rank], num_clients_list[rank-1])
        
        
class SynchronousServerManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.host_ip = 'localhost'
        self.model = CNN_Mnist()

    def test_init(self):
        self._check_init(mode="LOCAL")
        self._check_init(mode="GLOBAL")

    def _check_init(self, mode):
        port = '3444'
        network = DistNetwork(address=(self.host_ip, port),
                              world_size=1,
                              rank=0)
        handler = TestServerHandler(self.model, cuda=False)
        manager = SynchronousServerManager(network, handler, mode=mode)
        self.assertIsInstance(manager._handler, ServerHandler)
        self.assertEqual(manager.mode, mode)
