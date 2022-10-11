import unittest
import time
from copy import deepcopy

from fedlab.core.client.manager import ClientManager, ActiveClientManager, PassiveClientManager
from fedlab.core.client import ORDINARY_TRAINER, SERIAL_TRAINER
from fedlab.core.network import DistNetwork
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.core.client.trainer import ClientTrainer, SerialClientTrainer
from fedlab.utils import MessageCode, Logger
from fedlab.models import MLP

import torch
import torch.distributed as dist
from torch.multiprocessing import Process




if __name__ == "__main__":
    host_ip = 'localhost'
    port = '3333'
    server_network = DistNetwork(address=(host_ip, port),
                                 world_size=2,
                                 rank=0)
    server_network.init_network_connection()
    print(f"Server init done")
    # # receive setup package
    num_clients = 1
    _, message_code, contents = server_network.recv(src=num_clients)
    print(f"Server received from client-{num_clients-1}")
    # self.assertEqual(message_code, MessageCode.SetUp)
    # receive synchronization package
    # for i in range(num_clients):
    #     print(f"Server waiting for client-{i}'s package")
    #     _, message_code, content = server_network.recv(src=src)
    #     print(f"Server received from client-{i}")
    #     self.assertEqual(message_code, MessageCode.ParameterUpdate)
    #     self._check_content_eq(content, check_contents[i])
    # shutdown server network
    server_network.close_network_connection()