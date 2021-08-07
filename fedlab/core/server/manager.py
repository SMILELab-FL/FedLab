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
from queue import Queue
import logging

from ..network_manager import NetworkManager
from ..communicator.processor import Package, PackageProcessor
from ..network import DistNetwork
from ..server.handler import ParameterServerBackendHandler

from ...utils.serialization import SerializationTool
from ...utils.logger import logger
from ...utils.message_code import MessageCode

DEFAULT_SERVER_RANK = 0


class ServerSynchronousManager(NetworkManager):
    """Synchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronously communicate with clients following agreements defined in :meth:`run`.

    Args:
        handler (ParameterServerBackendHandler, optional): backend calculation class for parameter server.
        network (DistNetwork): object to manage torch.distributed network communication.
        com_round (int): the global round of FL iteration.
        logger (logger, optional): output cmd info to file.
    """
    def __init__(self, handler, network: DistNetwork, logger: logger = None):

        super(ServerSynchronousManager, self).__init__(network, handler)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def run(self):
        """Main Process"""
        self._LOGGER.info(
            "Initializing pytorch distributed group\n Waiting for connection requests from clients"
        )
        self._network.init_network_connection()
        self._LOGGER.info("Connect to clients successfully")

        while self._handler.stop_condition():
            self.activate_clients()
            while True:
                sender, message_code, payload = PackageProcessor.recv_package()
                if self.on_receive(sender, message_code, payload):
                    break

        self.shutdown_clients()
        self._network.close_network_connection()

    def on_receive(self, sender, message_code, payload):
        """Communication agreements of synchronous FL.

        - Server receive parameter from client. Transmit to handler for aggregation.

        Args:
            sender (int): rank of sender process.
            message_code (:class:`fedlab_utils.message_code.MessageCode`): message code
            payload (list[torch.Tensor]): list of tensors.

        Raises:
            Exception: Un expected MessageCode.

        Returns:
            [type]: [description]
        """
        if message_code == MessageCode.ParameterUpdate:
            model_parameters = payload[0]
            update_flag = self._handler.add_single_model(
                sender, model_parameters)
            return update_flag
        else:
            raise Exception("Unexpected message code {}".format(message_code))

    def activate_clients(self):
        """Activate some of clients to join this FL round"""
        clients_this_round = self._handler.sample_clients()
        self._LOGGER.info(
            "client id list for this FL round: {}".format(clients_this_round))

        for client_idx in clients_this_round:
            model_params = self._handler.model
            pack = Package(message_code=MessageCode.ParameterUpdate,
                           content=model_params)
            PackageProcessor.send_package(pack, dst=client_idx)

    def shutdown_clients(self):
        """Shutdown all clients"""
        for client_idx in range(1, self._handler.client_num_in_total + 1):
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=client_idx)


class ServerAsynchronousManager(NetworkManager):
    """Asynchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Asynchronously communicate with clients following agreements defined in :meth:`run`.

    Args:
        handler (ParameterServerBackendHandler, optional): backend calculation class for parameter server.
        network (DistNetwork): object to manage torch.distributed network communication.
        logger (logger, optional): output cmd info to file.
    """
    def __init__(self, handler, network: DistNetwork, logger: logger = None):

        super(ServerAsynchronousManager, self).__init__(network, handler)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        self.message_queue = Queue()

    def run(self):
        """Main process"""
        self._LOGGER.info(
            "Initializing pytorch distributed group \nWaiting for connection requests from clients"
        )
        self._network.init_network_connection()
        self._LOGGER.info("Connect to clients successfully")

        watching = threading.Thread(target=self.watching_queue)
        watching.start()

        while self._handler.stop_condition():
            sender, message_code, payload = PackageProcessor.recv_package()
            self.on_receive(sender, message_code, payload)

        self.shutdown_clients()
        self._network.close_network_connection()

    def on_receive(self, sender, message_code, payload):
        """Communication agreements of asynchronous FL.

        - Server receive ParameterRequest from client. Send model parameter to client. 
        - Server receive ParameterUpdate from client. Transmit parameters to queue waiting for aggregation.

        Args:
            sender (int): rank of sender process.
            message_code (:class:`fedlab_utils.message_code.MessageCode`): message code
            payload (list[torch.Tensor]): list of tensors.

        Raises:
            ValueError: [description]
        """
        if message_code == MessageCode.ParameterRequest:
            pack = Package(message_code=MessageCode.ParameterUpdate)
            model_params = self._handler.model
            pack.append_tensor_list([model_params, self._handler.global_time])
            self._LOGGER.info(
                "Send model to rank {}, the model current updated time {}".
                format(sender, int(self._handler.global_time.item())))
            PackageProcessor.send_package(pack, dst=sender)

        elif message_code == MessageCode.ParameterUpdate:
            self.message_queue.put((sender, message_code, payload))

        else:
            raise ValueError("Unexpected message code {}".format(message_code))

    def watching_queue(self):
        """Asynchronous communication maintain a message queue. A new thread will be started to run this function.

            Note:
                Customize strategy by overwrite this function.
        """
        while self._handler.stop_condition():
            _, _, payload = self.message_queue.get()
            parameters = payload[0]
            model_time = payload[1]
            self._handler.update_model(parameters, model_time)

    def shutdown_clients(self):
        """Shutdown all clients"""
        for client_idx in range(1, self._handler.client_num_in_total + 1):
            _, message_code, _ = PackageProcessor.recv_package(src=client_idx)
            if message_code == MessageCode.ParameterUpdate:
                PackageProcessor.recv_package(
                    src=client_idx)  # the next package is model request
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=client_idx)
