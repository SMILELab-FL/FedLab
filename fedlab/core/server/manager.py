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

from ..network_manager import NetworkManager
from ..coordinator import Coordinator
from ...utils import Logger, MessageCode

DEFAULT_SERVER_RANK = 0


class ServerManager(NetworkManager):
    """Base class of ServerManager.

    Args:
        network (DistNetwork): Network configuration and interfaces.
        handler (ParameterServerBackendHandler): Performe global server aggregation procedure.
    """

    def __init__(self, network, handler):
        super().__init__(network)
        self._handler = handler
        self.coordinator = None

    def setup(self):
        """Initialization Stage.

        - Server accept local client num report from client manager.
        - Init a coordinator for client_id -> rank mapping.
        """
        super().setup()
        rank_client_id_map = {}

        for rank in range(1, self._network.world_size):
            _, _, content = self._network.recv(src=rank)
            rank_client_id_map[rank] = content[0].item()
        self.coordinator = Coordinator(rank_client_id_map)
        if self._handler is not None:
            self._handler.client_num_in_total = self.coordinator.total


class SynchronousServerManager(ServerManager):
    """Synchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronously communicate with clients following agreements defined in :meth:`main_loop`.

    Args:
        network (DistNetwork): Network configuration and interfaces.
        handler (ParameterServerBackendHandler): Backend calculation handler for parameter server.
        logger (Logger, optional): Object of :class:`Logger`.
    """

    def __init__(self, network, handler, logger=None):
        super(SynchronousServerManager, self).__init__(network, handler)
        self._LOGGER = Logger() if logger is None else logger

    def setup(self):
        return super().setup()

    def main_loop(self):
        """Actions to perform in server when receiving a package from one client.

        Server transmits received package to backend computation handler for aggregation or others
        manipulations.

        Loop:
            1. activate clients for current training round.
            2. listen for message from clients -> transmit received parameters to server backend.

        Note:
            Communication agreements related: user can overwrite this function to customize
            communication agreements. This method is key component connecting behaviors of
            :class:`ParameterServerBackendHandler` and :class:`NetworkManager`.

        Raises:
            Exception: Unexpected :class:`MessageCode`.
        """
        while self._handler.if_stop is not True:
            activate = threading.Thread(target=self.activate_clients)
            activate.start()

            while True:
                sender_rank, message_code, payload = self._network.recv()
                if message_code == MessageCode.ParameterUpdate:
                    if self._handler._update_global_model(payload):
                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))

    def shutdown(self):
        """Shutdown stage."""
        self.shutdown_clients()
        super().shutdown()

    def activate_clients(self):
        """Activate subset of clients to join in one FL round

        Manager will start a new thread to send activation package to chosen clients' process rank.
        The ranks of clients are obtained from :meth:`handler.sample_clients`.
        """
        self._LOGGER.info("Client activation procedure")
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self._LOGGER.info("Client id list: {}".format(clients_this_round))

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(content=[id_list] + downlink_package,
                               message_code=MessageCode.ParameterUpdate,
                               dst=rank)

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to each client with :attr:`MessageCode.Exit`.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.
        """
        client_list = range(self._handler.client_num_in_total)
        rank_dict = self.coordinator.map_id_list(client_list)

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(content=[id_list] + downlink_package,
                               message_code=MessageCode.Exit,
                               dst=rank)

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(src=self._network.world_size -
                                                1)
        assert message_code == MessageCode.Exit


class AsynchronousServerManager(ServerManager):
    """Asynchronous communication network manager for server

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Asynchronously communicate with clients following agreements defined in :meth:`run`.

    Args:
        network (DistNetwork): Network configuration and interfaces.
        handler (ParameterServerBackendHandler): Backend computation handler for parameter server.
        logger (Logger, optional): Object of :class:`Logger`.
    """

    def __init__(self, network, handler, logger=None):
        super(AsynchronousServerManager, self).__init__(network, handler)
        self._LOGGER = Logger() if logger is None else logger

        self.message_queue = Queue()

    def setup(self):
        return super().setup()

    def main_loop(self):
        """Communication agreements of asynchronous FL.

        - Server receive ParameterRequest from client. Send model parameter to client.
        - Server receive ParameterUpdate from client. Transmit parameters to queue waiting for aggregation.

        Raises:
            ValueError: invalid message code.
        """
        updater = threading.Thread(target=self.updater_thread, daemon=True)
        updater.start()

        while self._handler.if_stop is not True:
            sender, message_code, payload = self._network.recv()

            if message_code == MessageCode.ParameterRequest:
                self._network.send(content=self._handler.downlink_package,
                                   message_code=MessageCode.ParameterUpdate,
                                   dst=sender)

            elif message_code == MessageCode.ParameterUpdate:
                self.message_queue.put((sender, message_code, payload))

            else:
                raise ValueError(
                    "Unexpected message code {}.".format(message_code))

    def shutdown(self):
        self.shutdown_clients()
        super().shutdown()

    def updater_thread(self):
        """Asynchronous communication maintain a message queue. A new thread will be started to run this function."""
        while self._handler.if_stop is not True:
            _, message_code, payload = self.message_queue.get()
            self._handler._update_global_model(payload)

            assert message_code == MessageCode.ParameterUpdate

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to clients with ``MessageCode.Exit``.
        """
        for rank in range(1, self._network.world_size):
            _, message_code, _ = self._network.recv(src=rank)  # client request
            if message_code == MessageCode.ParameterUpdate:
                self._network.recv(
                    src=rank
                )  # the next package is model request, which is ignored in shutdown stage.
            self._network.send(message_code=MessageCode.Exit, dst=rank)

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(src=self._network.world_size -
                                                1)
        assert message_code == MessageCode.Exit
