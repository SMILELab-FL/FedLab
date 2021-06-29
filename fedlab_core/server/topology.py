import threading
import torch
from queue import Queue
import logging

from abc import ABC, abstractmethod
import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab_utils.serialization import SerializationTool
from fedlab_core.communicator.processor import Package, PackageProcessor, MessageCode

DEFAULT_SERVER_RANK = 0

class ServerBasicTopology(Process, ABC):
    """Abstract class for server network topology

    If you want to define your own topology agreements, please subclass it.

    Args:
        server_address (tuple): Address of server in form of ``(SERVER_ADDR, SERVER_IP)``
        dist_backend (str): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
        and ``nccl``
    """
    def __init__(self, handler, server_address, dist_backend='gloo'):
        self._handler = handler
        self.server_address = server_address
        self.dist_backend = dist_backend

    @abstractmethod
    def run(self):
        """Main process, define your server's behavior"""
        raise NotImplementedError()

    @abstractmethod
    def listen_clients(self):
        """Listen messages from clients"""
        raise NotImplementedError()
    
    def activate_clients(self):
        """Activate some of clients to join this FL round"""
        raise NotImplementedError()

    def init_network_connection(self, world_size):
        dist.init_process_group(backend=self.dist_backend,
                                init_method='tcp://{}:{}'.format(
                                    self.server_address[0],
                                    self.server_address[1]),
                                rank=DEFAULT_SERVER_RANK,
                                world_size=world_size)

    def shutdown_clients(self):
        """Shutdown all clients"""
        for client_idx in range(1, self._handler.client_num_in_total + 1):
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=client_idx)


class ServerSynchronousTopology(ServerBasicTopology):
    """Synchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronize with clients following agreements defined in :meth:`run`.

    Args:
        server_handler: Subclass of :class:`ParameterServerHandler`
        server_address (tuple): Address of this server in form of ``(SERVER_ADDR, SERVER_IP)``
        dist_backend (str or Backend): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
        and ``nccl``. Default: ``"gloo"``
        logger （`logger`, optional）:
    """

    def __init__(self,
                 handler,
                 server_address,
                 dist_backend="gloo",
                 logger=None):

        super(ServerSynchronousTopology,
              self).__init__(handler=handler,
                             server_address=server_address,
                             dist_backend=dist_backend)

        self._LOGGER = logging if logger is None else logger
        self._LOGGER.info(
            "Server initializes with ip address {}:{} and distributed backend {}"
            .format(server_address[0], server_address[1], dist_backend))

        self.global_round = 3  # for current test
        # TODO 考虑通过pytorch.kv_store实现，client参数请求

    def run(self):
        """Main Process"""
        self._LOGGER.info(
            "Initializing pytorch distributed group\n Waiting for connection requests from clients"
        )
        self.init_network_connection(
            world_size=self._handler.client_num_in_total + 1)
        self._LOGGER.info("Connect to clients successfully")

        for round_idx in range(self.global_round):
            self._LOGGER.info("Global FL round {}/{}".format(
                round_idx + 1, self.global_round))

            activate = threading.Thread(target=self.activate_clients)
            listen = threading.Thread(target=self.listen_clients)

            activate.start()
            listen.start()

            activate.join()
            listen.join()

        self.shutdown_clients()

    def activate_clients(self):
        """Activate some of clients to join this FL round"""
        clients_this_round = self._handler.select_clients()
        self._LOGGER.info(
            "client id list for this FL round: {}".format(clients_this_round))

        for client_idx in clients_this_round:
            model_params = SerializationTool.serialize_model(
                self._handler.model)
            pack = Package(message_code=MessageCode.ParameterUpdate, content=model_params)
            PackageProcessor.send_package(pack, dst=client_idx)

    def listen_clients(self):
        """Listen messages from clients"""
        self._handler.train()  # turn train_flag to True
        # server_handler will turn off train_flag once the global model is updated
        while self._handler.train_flag:
            sender, message_code, payload = PackageProcessor.recv_package()
            if message_code == MessageCode.ParameterUpdate:
                self._handler.on_receive(sender, message_code, payload)
            else:
                self._LOGGER.info(
                    "invalid message code {}".format(message_code))

class ServerAsynchronousTopology(ServerBasicTopology):
    """Asynchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Asynchronize with clients following agreements defined in :meth:`run`.

    Args:
        server_handler: Subclass of :class:`ParameterServerHandler`
        server_address (tuple): Address of this server in form of ``(SERVER_ADDR, SERVER_IP)``
        dist_backend (str or Backend): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
        and ``nccl``. Default: ``"gloo"``
        logger （`logger`, optional）:
    """
    def __init__(self,
                 handler,
                 server_address,
                 dist_backend="gloo",
                 logger=None):

        super(ServerAsynchronousTopology,
              self).__init__(handler=handler,
                             server_address=server_address,
                             dist_backend=dist_backend)

        self._LOGGER = logging if logger is None else logger
        self._LOGGER.info(
            "Server initializes with ip address {}:{} and distributed backend {}"
            .format(server_address[0], server_address[1], dist_backend))
        self.total_update_num = 5  # control server to end receiving msg

    def run(self):
        """Main process"""
        self._LOGGER.info("Initializing pytorch distributed group")
        self._LOGGER.info("Waiting for connection requests from clients")
        self.init_network_connection(
            world_size=self._handler.client_num_in_total + 1)
        self._LOGGER.info("Connect to clients successfully")

        listen = threading.Thread(target=self.listen_clients)

        listen.start()

        listen.join()
        self.shutdown_clients()

    def listen_clients(self):
        """Listen messages from clients"""
        current_update_time = torch.zeros(1) # TODO: current_update_time应该由handler更新
        while current_update_time < self.total_update_num:
            sender, message_code, content = PackageProcessor.recv_package()
            self._LOGGER.info("Package received from {}, message code {}".format(sender, message_code))
            # 有关模型和联邦优化算法的处理都放到handler里实现，topology只负责处理网络通信
            # 如果是参数请求，则返回模型信息
            # 如果是模型上传更新，则将信息传到handler处理（调用self._handler.on_receive()
            if message_code == MessageCode.ParameterRequest:
                self._LOGGER.info("Send model to rank {}, the model current updated time {}".format(
                        sender, int(current_update_time.item())))
                pack = Package(message_code=MessageCode.ParameterUpdate)
                model_params = SerializationTool.serialize_model(self._handler.model)
                pack.append_tensor_list([model_params, self._handler.model_update_time])
                PackageProcessor.send_package(pack, dst=sender)
            elif message_code == MessageCode.ParameterUpdate:
                self._handler.on_receive(sender, message_code, content)
                current_update_time += 1
                self._handler.model_update_time = current_update_time
            else:
                raise ValueError("Invalid message code {}".format(message_code))
        
        self._LOGGER.info("{} times model update are completed".format(self.total_update_num))

    def shutdown_clients(self):
        """Shutdown all clients"""
        for client_idx in range(1, self._handler.client_num_in_total + 1):
            # deal the remaining package, end communication
            sender, message_code, payload = PackageProcessor.recv_package(src=client_idx)
            # for model request, end directly; for remaining model update, get the next model request package to end
            if message_code == MessageCode.ParameterUpdate:
                PackageProcessor.recv_package(src=client_idx)  # the next package is model request
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=client_idx)