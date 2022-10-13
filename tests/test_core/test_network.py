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

    def test_broadcast_send_and_recv(self):
        port = "5555"
        world_size = 6
        sender_num = 3
        recv_num = world_size - sender_num

        sender_ranks = list(range(sender_num))
        recv_ranks = list(range(sender_num, world_size))

        rank_nets = []
        processes = []

        # set DistNetwork for each rank
        for rank in range(world_size):
            rank_net = DistNetwork(address=(self.host_ip, port),
                                   world_size=world_size,
                                   rank=rank)
            rank_nets.append(rank_net)

        # add sender processes
        for rank in sender_ranks:
            p = Process(target=self._broadcast_sender_run, 
                        args=(rank_nets[rank], self.tensor_list, recv_ranks))
            processes.append(p)
        # add receiver processes
        for rank in recv_ranks:
            p = Process(target=self._broadcast_receiver_run,
                        args=(rank_nets[rank], self.tensor_list, sender_ranks))
            processes.append(p)
        # run the broadcast_send/recv
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def test_broadcast_send_and_recv_default_dst(self):
        port = "6666"
        world_size = 4

        sender_ranks = [3]
        recv_ranks = [0, 1, 2]

        rank_nets = []
        processes = []

        # set DistNetwork for each rank
        for rank in range(world_size):
            rank_net = DistNetwork(address=(self.host_ip, port),
                                   world_size=world_size,
                                   rank=rank)
            rank_nets.append(rank_net)

        # add sender processes
        for rank in sender_ranks:
            p = Process(target=self._broadcast_sender_run, 
                        args=(rank_nets[rank], self.tensor_list))
            processes.append(p)
        # add receiver processes
        for rank in recv_ranks:
            p = Process(target=self._broadcast_receiver_run,
                        args=(rank_nets[rank], self.tensor_list, sender_ranks))
            processes.append(p)
        # run the broadcast_send/recv
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def test_broadcast_send_and_recv_default_src(self):
        port = "7777"
        world_size = 4

        sender_ranks = [0, 1, 2]
        recv_ranks = [3]

        rank_nets = []
        processes = []

        # set DistNetwork for each rank
        for rank in range(world_size):
            rank_net = DistNetwork(address=(self.host_ip, port),
                                   world_size=world_size,
                                   rank=rank)
            rank_nets.append(rank_net)

        # add sender processes
        for rank in sender_ranks:
            p = Process(target=self._broadcast_sender_run, 
                        args=(rank_nets[rank], self.tensor_list, recv_ranks))
            processes.append(p)
        # add receiver processes
        for rank in recv_ranks:
            p = Process(target=self._broadcast_receiver_run,
                        args=(rank_nets[rank], self.tensor_list))
            processes.append(p)
        # run the broadcast_send/recv
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def test_broadcast_send_invalid_dst(self):
        port = "9999"
        rank_net = DistNetwork(address=(self.host_ip, port),
                                   world_size=1,
                                   rank=0)
        rank_net.init_network_connection()
        with self.assertRaises(TypeError):
            invalid_dst = 1
            rank_net.broadcast_send(self.tensor_list, 
                                    message_code=MessageCode.ParameterUpdate, 
                                    dst=invalid_dst)
        rank_net.close_network_connection()

    def test_broadcast_recv_invalid_src(self):
        port = "2222"
        rank_net = DistNetwork(address=(self.host_ip, port),
                                   world_size=1,
                                   rank=0)
        rank_net.init_network_connection()
        with self.assertRaises(TypeError):
            invalid_src = 1
            rank_net.broadcast_recv(self.tensor_list, 
                                    message_code=MessageCode.ParameterUpdate, 
                                    src=invalid_src)
        rank_net.close_network_connection()
            

    def _rank_run(self, rank_name, rank_net):
        rank_net.init_network_connection()
        time.sleep(30)
        rank_net.close_network_connection()

    def _check_pids_by_port(self, host_ip, port, rank_num, kind='tcp'):
        # find PIDs of processes using certain connection like 'tcp' with specific port number
        time.sleep(10)
        pids = []
        connects = psutil.net_connections(kind=kind)
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
        self.assertTrue(self._check_content_eq(content, check_content))
        recv_network.close_network_connection()

    def _broadcast_sender_run(self, send_network, content, dst=None):
        send_network.init_network_connection()
        send_network.broadcast_send(content, 
                                    message_code=MessageCode.ParameterUpdate, 
                                    dst=dst)
        send_network.close_network_connection()

    def _broadcast_receiver_run(self, recv_network, check_content, src=None):
        recv_network.init_network_connection()
        sender_ranks, _, contents = recv_network.broadcast_recv(src=src)
        # check sender rank
        if src is None:
            src = list(range(recv_network.world_size))
        if recv_network.rank in src:
            src.remove(recv_network.rank)
        self.assertEqual(sorted(src), sorted(sender_ranks)) 
        # check received contents 
        for idx, sender_rank in enumerate(sender_ranks):
            content = contents[idx]
            self.assertTrue(self._check_content_eq(content, check_content))
        recv_network.close_network_connection()

    def _check_content_eq(self, content, check_content):
        check_res = [torch.equal(t, p_t) for t, p_t in zip(content, check_content)]
        return all(check_res)



    
