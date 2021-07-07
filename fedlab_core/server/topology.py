import threading
import torch
from queue import Queue
import logging

from fedlab_core.topology import Topology
from fedlab_utils.serialization import SerializationTool
from fedlab_core.communicator.processor import Package, PackageProcessor, MessageCode
from fedlab_core.network import DistNetwork
from fedlab_utils.logger import logger

DEFAULT_SERVER_RANK = 0


class ServerSynchronousTopology(Topology):
    """Synchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronize with clients following agreements defined in :meth:`run`.

    Args:
        handler (`ClientBackendHandler` or `ParameterServerHandler`, optional): object to deal.
        newtork (`DistNetwork`): object to manage torch.distributed network communication.
        logger (`fedlab_utils.logger`, optional): Tools, used to output information.
    """
    def __init__(self, handler, network: DistNetwork, logger: logger = None):

        super(ServerSynchronousTopology, self).__init__(network, handler)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        self.global_round = 3  # for current test
        # TODO 考虑通过pytorch.kv_store实现，client参数请求

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
        clients_this_round = self._handler.select_clients()
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


class ServerAsynchronousTopology(Topology):
    """Asynchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Asynchronize with clients following agreements defined in :meth:`run`.

    Args:
        server_handler: Subclass of :class:`ParameterServerHandler`
        server_address (tuple): Address of this server in form of ``(SERVER_ADDR, SERVER_IP)``
        dist_backend (str or Backend): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
        and ``nccl``. Default: ``"gloo"``
        logger (`fedlab_utils.logger`, optional): Tools, used to output information.
    """
    def __init__(self, handler, network: DistNetwork, logger: logger = None):

        super(ServerAsynchronousTopology, self).__init__(handler, network)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        self.total_update_num = 5  # control server to end receiving msg
        self.message_queue = Queue()

    def run(self):
        """Main process"""
        self._LOGGER.info("Initializing pytorch distributed group")
        self._LOGGER.info("Waiting for connection requests from clients")
        self._network.init_network_connection()
        self._LOGGER.info("Connect to clients successfully")

        current_time = 0
        watching = threading.Thread(target=self.watching_queue)
        watching.start()

        while current_time < self.total_update_num:
            sender, message_code, payload = PackageProcessor.recv_package()
            self.on_receive(sender, message_code, payload)
            current_time += 1

        self.shutdown_clients()

    def on_receive(self, sender, message_code, payload):
        if message_code == MessageCode.ParameterRequest:
            pack = Package(message_code=MessageCode.ParameterUpdate)
            model_params = SerializationTool.serialize_model(
                self._handler.model)
            pack.append_tensor_list(
                [model_params, self._handler.model_update_time])
            self._LOGGER.info(
                "Send model to rank {}, the model current updated time {}".
                format(sender, int(self._handler.model_update_time.item())))
            PackageProcessor.send_package(pack, dst=sender)

        elif message_code == MessageCode.ParameterUpdate:
            self.message_queue.put((sender, message_code, payload))

        else:
            raise ValueError("Unexpected message code {}".format(message_code))

    def watching_queue(self):
        while True:
            _, _, payload = self.message_queue.get()
            parameters = payload[0]
            model_time = payload[1]
            self._handler.update_model(parameters, model_time)
            self.total_update_num += 1

    def shutdown_clients(self):
        """Shutdown all clients"""
        for client_idx in range(1, self._handler.client_num_in_total + 1):
            # deal the remaining package, end communication
            _, message_code, _ = PackageProcessor.recv_package(src=client_idx)
            # for model request, end directly; for remaining model update, get the next model request package to end
            if message_code == MessageCode.ParameterUpdate:
                PackageProcessor.recv_package(
                    src=client_idx)  # the next package is model request
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=client_idx)
