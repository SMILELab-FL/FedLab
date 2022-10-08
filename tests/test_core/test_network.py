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

import os
from random import randint
import sys
import unittest
import psutil
import time

from fedlab.core.communicator.package import Package
from fedlab.utils.message_code import MessageCode
from fedlab.core.network import DistNetwork
from fedlab.core.communicator.processor import PackageProcessor


import torch
import torch.distributed as dist
from torch.multiprocessing import Process


class NetworkTestCase(unittest.TestCase):
    def setUp(self):
        tensor_num = 5
        tensor_sizes = [randint(3, 15) for _ in range(tensor_num)]
        self.tensor_list = [torch.rand(size) for size in tensor_sizes]
        self.host_ip = 'localhost'

    def test_single_distnetwork_default_ethernet(self):
        port = '3456'
        network = DistNetwork(address=(self.host_ip, port),
                              world_size=1,
                              rank=0)
        network.init_network_connection()
        network.close_network_connection()

    def test_multiple_distnetwork(self):
        rank_num = 5
        rank_nets = []
        port = "4567"

        for rank in range(rank_num):
            rank_name = f"rank-{rank}"
            rank_net = DistNetwork(address=(self.host_ip, port),
                                            world_size=rank_num,
                                            rank=rank)
            rank_nets.append(rank_net)
        
        processes = []
        for rank in range(rank_num):
            rank_name = f"rank-{rank}"
            processes.append(
                Process(target=self._rank_run,  args=(rank_name, rank_nets[rank]))
            )
        
        processes.append(
            Process(target=self._check_pids_by_port, args=(self.host_ip, port, rank_num))
        )

        for p in processes:
            p.start()
        
        for p in processes:
            p.join()

    def test_send_and_recv(self):
        port = "3333"
        send_network = DistNetwork(address=(self.host_ip, port),
                                   world_size=2,
                                   rank=1)
        recv_network = DistNetwork(address=(self.host_ip, port),
                                   world_size=2,
                                   rank=0)
        sender = Process(target=self._sender_run, args=(send_network, self.tensor_list))
        receiver = Process(target=self._receiver_run, args=(recv_network, self.tensor_list))
        
        sender.start()
        receiver.start()

        sender.join()
        receiver.join()

    def _rank_run(self, rank_name, rank_net):
        rank_net.init_network_connection()
        time.sleep(30)
        rank_net.close_network_connection()

    def _check_pids_by_port(self, host_ip, port, rank_num, kind='tcp'):
        time.sleep(10)
        pids = []
        connects = psutil.net_connections(kind=kind)
        # print(connects)
        for con in connects:
            if con.pid is not None:
                if con.laddr != tuple():
                    if str(con.laddr.port) == port:
                        pids.append(con.pid)
                if con.raddr != tuple():
                    if str(con.raddr.port) == port:
                        pids.append(con.pid)
        pids = set(pids)
        self.assertEqual(len(pids), rank_num)

    def _sender_run(self, send_network, content):
        send_network.init_network_connection()
        send_network.send(content, MessageCode.ParameterUpdate,dst=0)
        send_network.close_network_connection()

    def _receiver_run(self, recv_network, check_content):
        recv_network.init_network_connection()
        _, _, content = recv_network.recv(src=1)
        for t, p_t in zip(content, check_content):
            self.assertTrue(torch.equal(t, p_t))
        recv_network.close_network_connection()



    
