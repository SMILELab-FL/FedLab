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
import os
from random import randint

import sys

# sys.path.append("../../")

from fedlab.core.communicator.package import Package
from fedlab.utils.message_code import MessageCode
from fedlab.core.network import DistNetwork
from fedlab.core.communicator.processor import PackageProcessor

import torch
import torch.distributed as dist
from torch.multiprocessing import Process


# class test_sender(Process):
#     def __init__(self, content) -> None:
#         super(test_sender, self).__init__()
#         self.net = DistNetwork(address=("localhost", "3001"),
#                                world_size=2,
#                                rank=1)
#         self.tensor_list = content

#     def run(self):
#         self.net.init_network_connection()
#         p = Package(message_code=MessageCode.ParameterUpdate,
#                     content=self.tensor_list)
#         PackageProcessor.send_package(p, dst=0)
#         self.net.close_network_connection()

# def sender_run(network, content):
#     network.init_network_connection()
#     p = Package(message_code=MessageCode.ParameterUpdate,
#                 content=content)
#     PackageProcessor.send_package(p, dst=0)
#     network.close_network_connection()

# def receiver_run(network, check_content):
#     network.init_network_connection()
#     _, _, content = PackageProcessor.recv_package(src=1)
    

# class test_receiver(Process):
#     def __init__(self, check_content) -> None:
#         super(test_receiver, self).__init__()
#         self.net = DistNetwork(address=("localhost", "3001"),
#                                world_size=2,
#                                rank=0)
#         self.check_list = check_content

#     def run(self):
#         self.net.init_network_connection()
#         _, _, content = PackageProcessor.recv_package(src=1)

#         for t, p_t in zip(content, self.check_list):
#             assert torch.equal(t, p_t)

#         self.net.close_network_connection()


class PackageProcessorTestCase(unittest.TestCase):
    def setUp(self):
        tensor_num = 5
        tensor_sizes = [randint(3, 15) for _ in range(tensor_num)]
        self.tensor_list = [torch.rand(size) for size in tensor_sizes]

    def test_send_and_receive(self):
        send_network = DistNetwork(address=("localhost", "3001"),
                                   world_size=2,
                                   rank=1)
        recv_network = DistNetwork(address=("localhost", "3001"),
                                   world_size=2,
                                   rank=0)
        sender = Process(target=self._sender_run, args=(send_network, self.tensor_list))
        receiver = Process(target=self._receiver_run, args=(recv_network, self.tensor_list))
        
        sender.start()
        receiver.start()

        sender.join()
        receiver.join()

    def _sender_run(self, send_network, content):
        send_network.init_network_connection()
        p = Package(message_code=MessageCode.ParameterUpdate,
                    content=content)
        PackageProcessor.send_package(p, dst=0)
        send_network.close_network_connection()

    def _receiver_run(self, recv_network, check_content):
        recv_network.init_network_connection()
        _, _, content = PackageProcessor.recv_package(src=1)
        # print(f"Length of content: {len(content)}")
        # print(f"Recieved Contents: {content}")
        for t, p_t in zip(content, check_content):
            self.assertTrue(torch.equal(t, p_t))
        recv_network.close_network_connection()
