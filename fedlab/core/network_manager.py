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

from torch.multiprocessing import Process
from .network import DistNetwork
from fedlab.core.client.trainer import ClientTrainer
from fedlab.core.server.handler import ParameterServerBackendHandler
from fedlab.utils.message_code import MessageCode

class NetworkManager(Process):
    """Abstract class

    Args:
        newtork (DistNetwork): object to manage torch.distributed network communication.
    """

    def __init__(self, network):
        super(NetworkManager, self).__init__()
        self._network = network

    def run(self):
        raise NotImplementedError()

    def on_receive(self, sender, message_code, payload):
        """Define the action to take when receiving a package.

        Args:
            sender (int): rank of current process.
            message_code (MessageCode): message code
            payload (torch.Tensor): list[torch.Tensor]
        """
        raise NotImplementedError()

    def setup(self):
        """Initialize network connection and necessary setups.

        Note:
            At first, ``self._network.init_network_connection()`` is required to be called.
            Overwrite this method to implement system setup message communication procedure.
        """
        self._network.init_network_connection()
