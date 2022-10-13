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
from io import StringIO
import unittest
from unittest.mock import patch
import psutil

from fedlab.core.network_manager import NetworkManager
from fedlab.core.network import DistNetwork

class ManagerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.host_ip = 'localhost'

    def test_setup_and_shutdown(self):
        port = '3456'
        network = DistNetwork(address=(self.host_ip, port),
                              world_size=1,
                              rank=0)
        manager = NetworkManager(network)
        self.assertIsInstance(manager._network, DistNetwork)

        manager.setup()
        network_num_after_setup = self._check_pids_by_port(self.host_ip, port)
        self.assertEqual(network_num_after_setup, 1)
        
        manager.shutdown()
        network_num_after_shutdown = self._check_pids_by_port(self.host_ip, port)
        self.assertEqual(network_num_after_shutdown, 0)

    def test_main_loop(self):
        port = '3444'
        network = DistNetwork(address=(self.host_ip, port),
                              world_size=1,
                              rank=0)
        manager = NetworkManager(network)
        with self.assertRaises(NotImplementedError):
            manager.main_loop()

    def test_run(self):
        class TestNetworkManager(NetworkManager):
            def __init__(self, network):
                super(TestNetworkManager, self).__init__(network)
            
            def main_loop(self):
                print("Main Loop done")

        port = '3457'
        network = DistNetwork(address=(self.host_ip, port),
                              world_size=1,
                              rank=0)
        manager = TestNetworkManager(network)
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            expected_output = "Main Loop done\n"
            manager.run()
            self.assertEqual(mock_stdout.getvalue(), expected_output)


    def _check_pids_by_port(self, host_ip, port, kind='tcp'):
        # find PIDs of processes using certain connection like 'tcp' with specific port number
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
        return len(pids)

