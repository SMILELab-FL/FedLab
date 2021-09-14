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

import threading
import torch
from torch.multiprocessing import Queue
import logging

from ..network_manager import NetworkManager
from ..communicator.processor import Package, PackageProcessor
from ..network import DistNetwork
from ..server.handler import ParameterServerBackendHandler
from ..coordinator import Coordinator

from ...utils.logger import Logger
from ...utils.message_code import MessageCode

DEFAULT_SERVER_RANK = 0


class ServerManager(NetworkManager):
    """Server Manager accept a object of DistNetwork and a ParameterServerBackendHandler

    Args:
        network (DistNetwork): network configuration.
        trainer (ParameterServerBackendHandler): performe global server aggregation procedure.
    """
    def __init__(self, network, handler):
        super().__init__(network)
        self._handler = handler
        
    def setup(self):
        """Setup agreements. Server accept local client num report from client manager, and generate coordinator."""
        super().setup()
        rank_client_id_map = {}
        
        for rank in range(1, self._network.world_size):
            _, _, content = PackageProcessor.recv_package(src=rank)
            rank_client_id_map[rank] = content[0].item()
        self.coordinator = Coordinator(rank_client_id_map)
        if self._handler is not None:
            self._handler.client_num_in_total = self.coordinator.total

class ServerSynchronousManager(ServerManager):
    """Synchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronously communicate with clients following agreements defined in :meth:`run`.

    Args:
        network (DistNetwork): Manage ``torch.distributed`` network communication.
        handler (ParameterServerBackendHandler, optional): Backend calculation handler for parameter server.
        logger (Logger, optional): :attr:`logger` for server handler. If set to ``None``, none logging output files will be generated while only on screen. Default: ``None``.
    """
    def __init__(self, network, handler, logger=None):

        super(ServerSynchronousManager, self).__init__(network, handler)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def run(self):
        """Main Process:
            1. Network initialization.

            2. Loop:
                2.1 activate clients.

                2.2 listen for message from clients -> transmit received parameters to server backend.

            3. Stop loop when stop condition is satisfied.

            4. Shut down clients, then close network connection.

        Note:
            user can overwrite this function to customize main process of Server.
        """

        self.setup()
        while self._handler.stop_condition() is not True:

            activate = threading.Thread(target=self.activate_clients)
            activate.start()

            # waiting for packages
            while True:
                sender, message_code, payload = PackageProcessor.recv_package()
                if self.on_receive(sender, message_code, payload):
                    break

        self.shutdown_clients()
        self._network.close_network_connection()

    def on_receive(self, sender, message_code, payload):
        """Actions to perform in server when receiving a package from one client.

        Server transmits received package to backend computation handler for aggregation or others
        manipulations.

        Note:
            Communication agreements related: user can overwrite this function to customize
            communication agreements. This method is key component connecting behaviors of
            :class:`ParameterServerBackendHandler` and :class:`NetworkManager`.

        Args:
            sender (int): Rank of sender client process.
            message_code (MessageCode): Predefined communication message code.
            payload (list[torch.Tensor]): A list of tensor, unpacked package received from clients.

        Raises:
            Exception: Unexpected :class:`MessageCode`.
        """
        if message_code == MessageCode.ParameterUpdate:
            model_parameters = payload[0]
            update_flag = self._handler.add_model(sender, model_parameters)
            return update_flag
        else:
            raise Exception("Unexpected message code {}".format(message_code))

    def activate_clients(self):
        """Activate subset of clients to join in one FL round

        Manager will start a new thread to send activation package to chosen clients' process rank.
        The ranks of clients are obtained from :meth:`handler.sample_clients`.

        Note:
            Communication agreements related: User can overwrite this function to customize
            activation package.
        """
        clients_this_round = self._handler.sample_clients()
        self._LOGGER.info(
            "client id list for this FL round: {}".format(clients_this_round))

        for client_id in clients_this_round:
            rank = client_id + 1

            model_parameters = self._handler.model_parameters  # serialized model params
            pack = Package(message_code=MessageCode.ParameterUpdate,
                           content=model_parameters)
            PackageProcessor.send_package(pack, dst=rank)

    def shutdown_clients(self):
        """Shut down all clients.

        Send package to every client with :attr:`MessageCode.Exit` to ask client to exit.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.

        """
        for rank in range(1, self._network.world_size):
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=rank)


class ServerAsynchronousManager(ServerManager):
    """Asynchronous communication network manager for server

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Asynchronously communicate with clients following agreements defined in :meth:`run`.

    Args:
        network (DistNetwork): Manage ``torch.distributed`` network communication.
        handler (ParameterServerBackendHandler, optional): Backend computation handler for parameter server.
        logger (Logger, optional): :attr:`logger` for server handler. If set to ``None``, none logging output files will be generated while only in console. Default: ``None``.
    """
    def __init__(self, network, handler, logger=None):

        super(ServerAsynchronousManager, self).__init__(network, handler)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        self.message_queue = Queue()

    def run(self):
        """Main process"""
        self.setup()

        watching = threading.Thread(target=self.watching_queue)
        watching.start()

        while self._handler.stop_condition() is not True:
            sender, message_code, payload = PackageProcessor.recv_package()
            self.on_receive(sender, message_code, payload)

        self.shutdown_clients()
        self._network.close_network_connection()

    def on_receive(self, sender, message_code, payload):
        """Communication agreements of asynchronous FL.

        - Server receive ParameterRequest from client. Send model parameter to client.
        - Server receive ParameterUpdate from client. Transmit parameters to queue waiting for aggregation.

        Args:
            sender (int): Rank of sender client process.
            message_code (MessageCode): message code
            payload (list[torch.Tensor]): List of tensors.

        Raises:
            ValueError: invalid message code.
        """
        if message_code == MessageCode.ParameterRequest:
            pack = Package(message_code=MessageCode.ParameterUpdate)
            model_parameters = self._handler.model_parameters
            pack.append_tensor_list(
                [model_parameters,
                 torch.Tensor(self._handler.server_time)])
            self._LOGGER.info(
                "Send model to rank {}, current server model time is {}".
                format(sender, self._handler.server_time))
            PackageProcessor.send_package(pack, dst=sender)

        elif message_code == MessageCode.ParameterUpdate:
            self.message_queue.put((sender, message_code, payload))

        else:
            raise ValueError("Unexpected message code {}".format(message_code))

    def watching_queue(self):
        """Asynchronous communication maintain a message queue. A new thread will be started to run this function.

        Note:
            Customize strategy by overwriting this function.
        """
        while self._handler.stop_condition() is not True:
            _, _, payload = self.message_queue.get()
            model_parameters = payload[0]
            model_time = payload[1]
            self._handler._update_model(model_parameters, model_time)

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to clients with ``MessageCode.Exit``.

        Note:
            Communication agreements related: user can overwrite this function to define close
            package.

        """
        for rank in range(1, self._network.world_size):
            _, message_code, _ = PackageProcessor.recv_package(src=rank)
            if message_code == MessageCode.ParameterUpdate:
                PackageProcessor.recv_package(
                    src=rank)  # the next package is model request
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=rank)