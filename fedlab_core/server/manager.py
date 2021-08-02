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

from fedlab_core.network_manager import NetworkManager
from fedlab_utils.serialization import SerializationTool
from fedlab_core.communicator.processor import Package, PackageProcessor
from fedlab_core.network import DistNetwork
from fedlab_utils.logger import logger
from fedlab_utils.message_code import MessageCode

DEFAULT_SERVER_RANK = 0


class ServerSynchronousManager(NetworkManager):
    """Synchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronize with clients following agreements defined in :meth:`run`.

    Args:
        handler (`ClientBackendHandler` or `ParameterServerHandler`, optional): object to deal.
        network (`DistNetwork`): object to manage torch.distributed network communication.
        logger (`fedlab_utils.logger`, optional): Tools, used to output information.
    """
    def __init__(self, handler, network: DistNetwork, logger: logger = None):

        super(ServerSynchronousManager, self).__init__(network, handler)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        self.global_round = 3  # for current test

    def run(self):
        """Main Process"""
        self._LOGGER.info(
            "Initializing pytorch distributed group\n Waiting for connection requests from clients"
        )
        self._network.init_network_connection()
        self._LOGGER.info("Connect to clients successfully")

        for round_idx in range(self.global_round):
            self._LOGGER.info("Global FL round {}/{}".format(
                round_idx + 1, self.global_round))

            activate = threading.Thread(target=self.activate_clients)
            activate.start()

            while True:
                sender, message_code, payload = PackageProcessor.recv_package()
                update_flag = self.on_receive(sender, message_code, payload)
                if update_flag:
                    break
        self.shutdown_clients()

    def on_receive(self, sender, message_code, payload):
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
            model_params = SerializationTool.serialize_model(
                self._handler.model)
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
    Asynchronize with clients following agreements defined in :meth:`run`.

    Args:
        handler (`ClientBackendHandler` or `ParameterServerHandler`, optional): object to deal.
        newtork (`DistNetwork`): object to manage torch.distributed network communication.
        logger (`fedlab_utils.logger`, optional): Tools, used to output information.
    """
    def __init__(self, handler, network: DistNetwork, logger: logger = None):

        super(ServerAsynchronousManager, self).__init__(network, handler)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        self.total_round = 5  # control server to end receiving msg
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

        while self._handler.global_time < self.total_round:
            sender, message_code, payload = PackageProcessor.recv_package()
            self.on_receive(sender, message_code, payload)

        self.shutdown_clients()

    def on_receive(self, sender, message_code, payload):
        if message_code == MessageCode.ParameterRequest:
            pack = Package(message_code=MessageCode.ParameterUpdate)
            model_params = SerializationTool.serialize_model(
                self._handler.model)
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
        while self._handler.global_time < self.total_round:

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
