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
from ..communicator.processor import Package, PackageProcessor
from ..coordinator import Coordinator

from ...utils.message_code import MessageCode
from ...utils import Logger

DEFAULT_SERVER_RANK = 0


class ServerManager(NetworkManager):
    """Base class of ServerManager.

    Args:
        network (DistNetwork): network configuration.
        handler (ParameterServerBackendHandler): performe global server aggregation procedure.
    """

    def __init__(self, network, handler):
        super().__init__(network)
        self._handler = handler
        self.coordinator = None

    def setup(self):
        """Initialization Stage. 
            
        - Server accept local client num report from client manager.
        - Init a coordinator for client_id mapping.
        """
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
    Synchronously communicate with clients following agreements defined in :meth:`main_loop`.

    Args:
        network (DistNetwork): Manage ``torch.distributed`` network communication.
        handler (ParameterServerBackendHandler): Backend calculation handler for parameter server.
        logger (Logger, optional): object of :class:`Logger`.
    """
    def __init__(self, network, handler, logger=Logger()):
        super(ServerSynchronousManager, self).__init__(network, handler)
        self._LOGGER = logger

    def shutdown(self):
        """Shutdown stage."""
        self.shutdown_clients()
        super().shutdown()

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
        while self._handler.stop_condition() is not True:
            activate = threading.Thread(target=self.activate_clients)
            activate.start()
            while True:
                sender, message_code, payload = PackageProcessor.recv_package()
                if message_code == MessageCode.ParameterUpdate:
                    model_parameters = payload[0]
                    if self._handler.add_model(sender, model_parameters):
                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))

    def activate_clients(self):
        """Activate subset of clients to join in one FL round

        Manager will start a new thread to send activation package to chosen clients' process rank.
        The ranks of clients are obtained from :meth:`handler.sample_clients`.
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
        """Shutdown all clients.

        Send package to each client with :attr:`MessageCode.Exit` to ask client to exit.

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
        logger (Logger, optional): object of :class:`Logger`.
    """
    def __init__(self, network, handler, logger=Logger()):
        super(ServerAsynchronousManager, self).__init__(network, handler)
        self._LOGGER = logger

        self.message_queue = Queue()

    def shutdown(self):
        self.shutdown_clients()
        super().shutdown()

    def main_loop(self):
        """Communication agreements of asynchronous FL.

        - Server receive ParameterRequest from client. Send model parameter to client.
        - Server receive ParameterUpdate from client. Transmit parameters to queue waiting for aggregation.

        Raises:
            ValueError: invalid message code.
        """
        watching = threading.Thread(target=self.watching_queue)
        watching.start()

        while self._handler.stop_condition() is not True:
            sender, message_code, payload = PackageProcessor.recv_package()

            if message_code == MessageCode.ParameterRequest:
                pack = Package(message_code=MessageCode.ParameterUpdate)
                model_parameters = self._handler.model_parameters
                pack.append_tensor_list([
                    model_parameters,
                    torch.Tensor(self._handler.server_time)
                ])
                self._LOGGER.info(
                    "Send model to rank {}, current server model time is {}".
                        format(sender, self._handler.server_time))
                PackageProcessor.send_package(pack, dst=sender)

            elif message_code == MessageCode.ParameterUpdate:
                self.message_queue.put((sender, message_code, payload))

            else:
                raise ValueError(
                    "Unexpected message code {}".format(message_code))

    def watching_queue(self):
        """Asynchronous communication maintain a message queue. A new thread will be started to run this function."""
        while self._handler.stop_condition() is not True:
            _, _, payload = self.message_queue.get()
            model_parameters = payload[0]
            model_time = payload[1]
            self._handler._update_model(model_parameters, model_time)

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to clients with ``MessageCode.Exit``.
        """
        for rank in range(1, self._network.world_size):
            _, message_code, _ = PackageProcessor.recv_package(src=rank)
            if message_code == MessageCode.ParameterUpdate:
                PackageProcessor.recv_package(
                    src=rank)  # the next package is model request
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=rank)
