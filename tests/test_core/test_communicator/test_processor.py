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

from fedlab.core.communicator.package import Package
from fedlab.utils.message_code import MessageCode
from fedlab.core.network import DistNetwork
from fedlab.core.communicator.processor import PackageProcessor

import torch
import torch.distributed as dist
from torch.multiprocessing import Process


"""
This test case uses multiprocessing! 
To enable code coverage for involved modules, need to to following steps:
   1. Set 
      ```
      concurrency = multiprocessing
      ```
      in .coveragerc config file
   2. Use following command to run coverage supporting for multi-process tests
      ```
      coverage run --concurrency=multiprocessing --parallel-mode setup.py test
      ```
   3. Combine coverage results from different processes using following command
      ```
      coverage combine
      ``` 
   4. Generate coverage report and generate xml report
      ```
      coverage report
      coverage xml
      ```
"""

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
        for t, p_t in zip(content, check_content):
            self.assertTrue(torch.equal(t, p_t))
        recv_network.close_network_connection()
