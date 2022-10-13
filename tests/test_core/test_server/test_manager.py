import unittest
import time
import random
from random import randint
from copy import deepcopy

import threading

from fedlab.core.client import ORDINARY_TRAINER, SERIAL_TRAINER
from fedlab.core.network import DistNetwork
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.core.server.handler import ServerHandler
from fedlab.core.server.manager import ServerManager, SynchronousServerManager, AsynchronousServerManager
from fedlab.utils import MessageCode, Logger, message_code

from ..task_setting_for_test import CNN_Mnist

import torch
from torch.multiprocessing import Queue
import torch.distributed as dist
from torch.multiprocessing import Process

class TestServerHandler(ServerHandler):
    def __init__(self, model, cuda=False, device=None, world_size=1, global_round=1):
        super().__init__(model, cuda, device)
        self.sample_ratio = 1.0
        self.global_round = global_round
        self.round = 0
        # client buffer
        self.client_buffer_cache = []
        self.world_size = world_size

    @property
    def downlink_package(self):
        return [torch.tensor([1,2,3,4])]

    @property
    def num_clients_per_round(self):
        return max(1, int(self.sample_ratio * self.num_clients))

    @property
    def if_stop(self):
        return self.round >= self.global_round

    def load(self, payload):
        self.client_buffer_cache.append(deepcopy(payload))
        assert len(self.client_buffer_cache) <= self.world_size

        if len(self.client_buffer_cache) == self.world_size:
            self.round += 1
            # reset cache
            self.client_buffer_cache = []
            return True
        else:
            return False

    def sample_clients(self):
        selection = random.sample(range(self.num_clients),
                                  self.num_clients_per_round)
        return sorted(selection)


class TestAsyncServerHandler(ServerHandler):
    def __init__(self, model, cuda=False, device=None, global_round=1):
        super().__init__(model, cuda, device)
        self.sample_ratio = 1.0
        self.global_round = global_round
        self.round = 0

    @property
    def downlink_package(self):
        return [torch.tensor([1,2,3,4]), torch.Tensor([self.round])]

    @property
    def num_clients_per_round(self):
        return max(1, int(self.sample_ratio * self.num_clients))

    @property
    def if_stop(self):
        return self.round >= self.global_round

    def load(self, payload):
        self.round += 1

    def sample_clients(self):
        selection = random.sample(range(self.num_clients),
                                  self.num_clients_per_round)
        return sorted(selection) 



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
        client_rank_num = 3
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

    def test_shutdown_clients(self):
        port = '5555'
        client_rank_num = 3
        num_clients_list = [randint(3,10) for _ in range(client_rank_num)]  # number of clients for each client trainer
        mode = "LOCAL"

        # set server network
        server_network = DistNetwork(address=(self.host_ip, port),
                                     world_size=1 + client_rank_num,
                                     rank=0)
        handler = TestServerHandler(self.model, cuda=False)
        server_manager = SynchronousServerManager(server_network, handler, mode=mode)
        server_p = Process(target=self._run_server_manager_shutdown_clients,
                           args=(server_manager,))
        

        client_networks = []
        client_proceses = []
        for rank in range(1, client_rank_num + 1):
            client_network = DistNetwork(address=(self.host_ip, port),
                                         world_size=1 + client_rank_num,
                                         rank=rank)
            client_networks.append(client_network)
            client_p = Process(target=self._run_client_network_shutdown,
                               args=(client_network, num_clients_list[rank-1], rank, client_rank_num+1))
            client_proceses.append(client_p)
        
        server_p.start()
        for client_p in client_proceses:
            client_p.start()

        server_p.join()
        for client_p in client_proceses:
            client_p.join()

    def test_shutdown(self):
        port = '6666'
        client_rank_num = 3
        num_clients_list = [randint(3,10) for _ in range(client_rank_num)]  # number of clients for each client trainer
        mode = "LOCAL"

        # set server network
        server_network = DistNetwork(address=(self.host_ip, port),
                                     world_size=1 + client_rank_num,
                                     rank=0)
        handler = TestServerHandler(self.model, cuda=False)
        server_manager = SynchronousServerManager(server_network, handler, mode=mode)
        server_p = Process(target=self._run_server_manager_shutdown,
                           args=(server_manager,))

        client_networks = []
        client_proceses = []
        for rank in range(1, client_rank_num + 1):
            client_network = DistNetwork(address=(self.host_ip, port),
                                         world_size=1 + client_rank_num,
                                         rank=rank)
            client_networks.append(client_network)
            client_p = Process(target=self._run_client_network_shutdown,
                               args=(client_network, num_clients_list[rank-1], rank, client_rank_num+1))
            client_proceses.append(client_p)
        
        server_p.start()
        for client_p in client_proceses:
            client_p.start()

        server_p.join()
        for client_p in client_proceses:
            client_p.join()

    def test_activate_clients(self):
        port = '7777'
        client_rank_num = 3
        num_clients_list = [randint(3,10) for _ in range(client_rank_num)]  # number of clients for each client trainer
        mode = "LOCAL"

        # set server network
        server_network = DistNetwork(address=(self.host_ip, port),
                                     world_size=1 + client_rank_num,
                                     rank=0)
        handler = TestServerHandler(self.model, cuda=False, world_size=client_rank_num+1)
        server_manager = SynchronousServerManager(server_network, handler, mode=mode)
        server_p = Process(target=self._run_server_manager_activate_clients,
                           args=(server_manager,))

        client_networks = []
        client_proceses = []
        for rank in range(1, client_rank_num + 1):
            client_network = DistNetwork(address=(self.host_ip, port),
                                         world_size=1 + client_rank_num,
                                         rank=rank)
            client_networks.append(client_network)
            client_p = Process(target=self._run_client_network_activate,
                               args=(client_network, num_clients_list[rank-1]))
            client_proceses.append(client_p)
        
        server_p.start()
        for client_p in client_proceses:
            client_p.start()

        server_p.join()
        for client_p in client_proceses:
            client_p.join()

    def test_main_loop(self):
        port = '2345'
        client_rank_num = 1
        num_clients_list = [randint(3,10) for _ in range(client_rank_num)]  # number of clients for each client trainer
        mode = "LOCAL"

        # set server network
        server_network = DistNetwork(address=(self.host_ip, port),
                                     world_size=1 + client_rank_num,
                                     rank=0)
        handler = TestServerHandler(self.model, cuda=False)
        server_manager = SynchronousServerManager(server_network, handler, mode=mode)
        server_p = Process(target=self._run_server_manager_main_loop,
                           args=(server_manager,))

        client_networks = []
        client_proceses = []
        for rank in range(1, client_rank_num + 1):
            client_network = DistNetwork(address=(self.host_ip, port),
                                         world_size=1 + client_rank_num,
                                         rank=rank)
            client_networks.append(client_network)
            client_p = Process(target=self._run_client_network_main_loop,
                               args=(client_network, num_clients_list[rank-1]))
            client_proceses.append(client_p)
        
        server_p.start()
        for client_p in client_proceses:
            client_p.start()

        server_p.join()
        for client_p in client_proceses:
            client_p.join()

    def _check_init(self, mode):
        port = '3444'
        network = DistNetwork(address=(self.host_ip, port),
                              world_size=1,
                              rank=0)
        handler = TestServerHandler(self.model, cuda=False)
        manager = SynchronousServerManager(network, handler, mode=mode)
        self.assertIsInstance(manager._handler, ServerHandler)
        self.assertEqual(manager.mode, mode)

    def _run_client_network_shutdown(self, client_network, num_clients, rank, world_size):
        # setup stage
        client_network.init_network_connection()
        client_network.send(content=torch.tensor(num_clients).int(),
                            message_code=MessageCode.SetUp,
                            dst=0)
        # receive Exit signal from server when server tries to shutdown clients
        _, message_code, _ = client_network.recv(src=0)
        assert message_code == MessageCode.Exit
        # send Exit signal back to server
        if rank == world_size - 1: 
            client_network.send(message_code=MessageCode.Exit,
                                dst=0)
        client_network.close_network_connection()

    def _run_client_network_activate(self, client_network, num_clients):
        # setup stage
        client_network.init_network_connection()
        client_network.send(content=torch.tensor(num_clients).int(),
                            message_code=MessageCode.SetUp,
                            dst=0)
        _, message_code, _ = client_network.recv(src=0)
        self.assertEqual(message_code, MessageCode.ParameterUpdate)
        client_network.close_network_connection()

    def _run_client_network_main_loop(self, client_network, num_clients):
        # setup stage
        client_network.init_network_connection()
        client_network.send(content=torch.tensor(num_clients).int(),
                            message_code=MessageCode.SetUp,
                            dst=0)
        # receive signal when server activates clients
        _, message_code, _ = client_network.recv(src=0)
        self.assertEqual(message_code, MessageCode.ParameterUpdate)
        # client sends ParameterUpdate signal to server
        client_network.send(message_code=MessageCode.ParameterUpdate, dst=0)
        # shutdown client manually
        client_network.close_network_connection()

    def _run_server_manager_shutdown_clients(self, server_manager):
        server_manager.setup()
        server_manager.shutdown_clients()
        server_manager._network.close_network_connection()  # close server network manually

    def _run_server_manager_shutdown(self, server_manager):
        server_manager.setup()
        server_manager.shutdown()

    def _run_server_manager_activate_clients(self, server_manager):
        server_manager.setup()
        server_manager.activate_clients()
        server_manager._network.close_network_connection()  # close server network manually

    def _run_server_manager_main_loop(self, server_manager):
        server_manager.setup()
        server_manager.main_loop()
        server_manager._network.close_network_connection()  # close server network manually


class AsynchronousServerManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.host_ip = 'localhost'
        self.port = '6666'
        self.model = CNN_Mnist()

    def test_init(self):
        network = DistNetwork(address=(self.host_ip, self.port),
                              world_size=1,
                              rank=0)
        handler = TestServerHandler(self.model, cuda=False)
        manager = AsynchronousServerManager(network, handler)
        self.assertIsInstance(manager._handler, ServerHandler)

    def test_shutdown_clients(self):
        client_rank_num = 3
        num_clients_list = [randint(3,10) for _ in range(client_rank_num)]  # number of clients for each client trainer
        mode = "LOCAL"

        # set server network
        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=1 + client_rank_num,
                                     rank=0)
        handler = TestServerHandler(self.model, cuda=False)
        server_manager = AsynchronousServerManager(server_network, handler)
        server_p = Process(target=self._run_server_manager_shutdown_clients,
                           args=(server_manager,))
        

        client_networks = []
        client_proceses = []
        for rank in range(1, client_rank_num + 1):
            client_network = DistNetwork(address=(self.host_ip, self.port),
                                         world_size=1 + client_rank_num,
                                         rank=rank)
            client_networks.append(client_network)
            client_p = Process(target=self._run_client_network_shutdown,
                               args=(client_network, num_clients_list[rank-1], rank, client_rank_num+1))
            client_proceses.append(client_p)
        
        server_p.start()
        for client_p in client_proceses:
            client_p.start()

        server_p.join()
        for client_p in client_proceses:
            client_p.join()

    def test_shutdown(self):
        client_rank_num = 3
        num_clients_list = [randint(3,10) for _ in range(client_rank_num)]  # number of clients for each client trainer
        mode = "LOCAL"

        # set server network
        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=1 + client_rank_num,
                                     rank=0)
        handler = TestServerHandler(self.model, cuda=False)
        server_manager = AsynchronousServerManager(server_network, handler)
        server_p = Process(target=self._run_server_manager_shutdown,
                           args=(server_manager,))
        

        client_networks = []
        client_proceses = []
        for rank in range(1, client_rank_num + 1):
            client_network = DistNetwork(address=(self.host_ip, self.port),
                                         world_size=1 + client_rank_num,
                                         rank=rank)
            client_networks.append(client_network)
            client_p = Process(target=self._run_client_network_shutdown,
                               args=(client_network, num_clients_list[rank-1], rank, client_rank_num+1))
            client_proceses.append(client_p)
        
        server_p.start()
        for client_p in client_proceses:
            client_p.start()

        server_p.join()
        for client_p in client_proceses:
            client_p.join()

    def test_updater_thread(self):
        server_rank = 0
        client_rank = 1

        # set server network
        server_network = DistNetwork(address=(self.host_ip, self.port),
                                     world_size=1,
                                     rank=server_rank)
        handler = TestServerHandler(self.model, cuda=False)
        server_manager = AsynchronousServerManager(server_network, handler)
        server_manager.message_queue.put((client_rank, MessageCode.ParameterUpdate, torch.tensor([1,2,3,4])))
        server_manager.updater_thread()

    # def test_main_loop(self):
    #     server_rank = 0
    #     client_rank = 1
    #     port = '7777'

    #     # set server network
    #     server_network = DistNetwork(address=(self.host_ip, port),
    #                                  world_size=2,
    #                                  rank=server_rank)
    #     handler = TestAsyncServerHandler(self.model, cuda=False)
    #     server_manager = AsynchronousServerManager(server_network, handler)
    #     server_p = Process(target=self._run_server_manager_main_loop,
    #                        args=(server_manager,))
        
    #     client_network = DistNetwork(address=(self.host_ip, port),
    #                                     world_size=2,
    #                                     rank=client_rank)
    #     client_p = Process(target=self._run_client_network_main_loop,
    #                         args=(client_network,))
        
    #     server_p.start()
    #     client_p.start()

    #     server_p.join()
    #     client_p.join()

    # def _run_client_network_main_loop(self, client_network):
    #     client_network.init_network_connection()
    #     # client sends request
    #     client_network.send(message_code=MessageCode.ParameterRequest, dst=0)
    #     _, message_code, _ = client_network.recv(src=0)
    #     self.assertEqual(message_code, MessageCode.ParameterUpdate)
    #     # client sends ParameterUpdate signal
    #     client_network.send(content=torch.tensor([1,2,3,4]), 
    #                         message_code=MessageCode.ParameterUpdate, 
    #                         dst=0)
    #     # empty signals: not sure how many signals are needed
    #     client_network.send(message_code=MessageCode.ParameterRequest, dst=0)
    #     _, message_code, _ = client_network.recv(src=0)
    #     # client_network.send(message_code=MessageCode.ParameterRequest, dst=0)
    #     # _, message_code, _ = client_network.recv(src=0)
    #     # client_network.send(message_code=MessageCode.ParameterRequest, dst=0)
    #     # _, message_code, _ = client_network.recv(src=0)
    #     print("ClientNetwork main loop done")
    #     # shutdown client
    #     client_network.close_network_connection()
    #     print("ClientNetwork shutdown done")

    # def _run_server_manager_main_loop(self, server_manager):
    #     server_manager._network.init_network_connection()
    #     server_manager.main_loop()
    #     server_manager._LOGGER.info("ServerManager main_loop done")
    #     server_manager._network.close_network_connection()
    #     server_manager._LOGGER.info("ServerManager shutdown down")

    def _run_server_manager_updater_thread(self, server_manager, client_rank):
        server_manager.message_queue.put((client_rank, MessageCode.ParameterUpdate, torch.tensor([1,2,3,4])))
        server_manager.updater_thread()

    def _run_server_manager_shutdown_clients(self, server_manager):
        server_manager.setup()
        server_manager.shutdown_clients()
        server_manager._network.close_network_connection()  # close server network manually

    def _run_client_network_shutdown(self, client_network, num_clients, rank, world_size):
        # setup stage
        client_network.init_network_connection()
        client_network.send(content=torch.tensor(num_clients).int(),
                            message_code=MessageCode.SetUp,
                            dst=0)
        client_network.send(message_code=MessageCode.ParameterUpdate,
                            dst=0)
        client_network.send(message_code=MessageCode.ParameterRequest,
                            dst=0)  # empty request but needed
        # receive Exit signal from server when server tries to shutdown clients
        _, message_code, _ = client_network.recv(src=0)
        assert message_code == MessageCode.Exit
        # send Exit signal back to server
        if rank == world_size - 1: 
            client_network.send(message_code=MessageCode.Exit,
                                dst=0)
        client_network.close_network_connection()

    def _run_server_manager_shutdown(self, server_manager):
        server_manager.setup()
        server_manager.shutdown()

    