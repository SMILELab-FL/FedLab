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

import logging

from ...utils.message_code import MessageCode
from ...utils.serialization import SerializationTool
from ...utils.logger import Logger

from ..communicator.processor import Package, PackageProcessor
from ..network import DistNetwork
from ..client.trainer import ClientTrainer
from ..network_manager import NetworkManager


class ClientPassiveManager(NetworkManager):
    """Passive communication :class:`NetworkManager` for client in synchronous FL

    Args:
        handler (ClientTrainer): Subclass of :class:`ClientTrainer`. Provides :meth:`train` and :attr:`model`.
        network (DistNetwork): Distributed network to use.
        logger (Logger, optional): object of :class:`Logger`.
    """
    def __init__(self, handler, network, logger=None):
        super(ClientPassiveManager, self).__init__(network, handler)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def run(self):
        """Main procedure of each client is defined here:
        1. client waits for data from server （PASSIVE）
        2. after receiving data, client will train local model
        3. client will synchronize with server actively
        """
        self._LOGGER.info("connecting with server")
        self.setup()

        while True: 
            self._LOGGER.info("Waiting for server...")
            # waits for data from server (default server rank is 0)
            sender_rank, message_code, payload = PackageProcessor.recv_package(
                src=0)
            # exit
            if message_code == MessageCode.Exit:
                self._LOGGER.info(
                    "Receive {}, Process exiting".format(message_code))
                self._network.close_network_connection()
                break
            else:
                # perform activation strategy
                self.on_receive(sender_rank, message_code, payload)

            # synchronize with server
            self.synchronize()

    def on_receive(self, sender_rank, message_code, payload):
        """Actions to perform when receiving new message, including local training

        Note:
            Customize the control flow of client corresponding with :class:`MessageCode`.

        Args:
            sender_rank (int): Rank of sender
            message_code (MessageCode): Agreements code defined in :class:`MessageCode`
            payload (list[torch.Tensor]): A list of tensors received from sender.
        """
        self._LOGGER.info("Package received from {}, message code {}".format(
            sender_rank, message_code))
        model_parameters = payload[0]
        self._handler.train(model_parameters=model_parameters)

    def setup(self):
        """Initialize network."""
        self._network.init_network_connection()

    def synchronize(self):
        """Synchronize local model with server actively

        Note:
            communication agreements related:
            Overwrite this function to customize package for synchronizing.
        """
        self._LOGGER.info("synchronize model parameters with server")
        model_parameters = self._handler.model_parameters
        pack = Package(message_code=MessageCode.ParameterUpdate,
                       content=model_parameters)
        PackageProcessor.send_package(pack, dst=0)


class ClientActiveManager(NetworkManager):
    """Active communication :class:`NetworkManager` for client in asynchronous FL

    Args:
        handler (ClientTrainer): Subclass of ClientTrainer. Provides :meth:`train` and :attr:`model`.
        network (DistNetwork): Distributed network to use.
        logger (Logger, optional): object of :class:`Logger`.
    """
    def __init__(self, handler, network, logger=None):
        super(ClientActiveManager, self).__init__(network, handler)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        self.model_time = None

    def run(self):
        """Main procedure of each client is defined here:
        1. client requests data from server (ACTIVE)
        2. after receiving data, client will train local model
        3. client will synchronize with server actively
        """

        self.setup()
        while True:
            self._LOGGER.info("Waiting for server...")
            # request model actively
            self._request_parameter()
            # waits for data from
            sender_rank, message_code, payload = PackageProcessor.recv_package(
                src=0)

            # exit
            if message_code == MessageCode.Exit:
                self._LOGGER.info(
                    "Recv {}, Process exiting".format(message_code))
                break

            # perform local training
            self.on_receive(sender_rank, message_code, payload)

            # synchronize with server
            self.synchronize()

    def on_receive(self, sender_rank, message_code, payload):
        """Actions to perform on receiving new message, including local training

        Args:
            sender_rank (int): Rank of sender
            message_code (MessageCode): Agreements code defined in: class:`MessageCode`.
            payload (list[torch.Tensor]): Received package, a list of tensors.
        """
        self._LOGGER.info("Package received from {}, message code {}".format(
            sender_rank, message_code))
        model_parameters = payload[0]
        self.model_time = payload[1]
        # move loading model params to the start of training
        self._handler.train(model_parameters=model_parameters)

    def setup(self):
        """Initialize network."""
        self._network.init_network_connection()

    def synchronize(self):
        """Synchronize local model with server actively"""
        self._LOGGER.info("synchronize procedure")
        model_parameters = self._handler.model_parameters
        pack = Package(message_code=MessageCode.ParameterUpdate)
        pack.append_tensor_list([model_parameters, self.model_time + 1])
        PackageProcessor.send_package(pack, dst=0)

    def _request_parameter(self):
        """send ParameterRequest"""
        self._LOGGER.info("request parameter procedure")
        pack = Package(message_code=MessageCode.ParameterRequest)
        PackageProcessor.send_package(pack, dst=0)




