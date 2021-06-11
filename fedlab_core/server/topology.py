import threading
import os
import torch
from queue import Queue
import logging

from abc import ABC, abstractmethod
import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab_utils.logger import logger
from fedlab_utils.serialization import SerializationTool
from fedlab_core.communicator.processor import Package, PackageProcessor, MessageCode


class ServerBasicTopology(Process, ABC):
    """Abstract class for server network topology

    If you want to define your own topology agreements, please subclass it.

    Args:
        server_address (tuple): Address of server in form of ``(SERVER_ADDR, SERVER_IP)``
        dist_backend (str): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
        and ``nccl``
    """
    def __init__(self, handler, server_address, dist_backend):
        self._handler = handler
        self.server_address = server_address
        self.dist_backend = dist_backend

    @abstractmethod
    def run(self):
        """Main process, define your server's behavior"""
        raise NotImplementedError()

    @abstractmethod
    def activate_clients(self):
        """Activate some of clients to join this FL round"""
        raise NotImplementedError()

    @abstractmethod
    def listen_clients(self):
        """Listen messages from clients"""
        raise NotImplementedError()

    def init_network_connection(self, world_size):
        dist.init_process_group(backend=self.dist_backend,
                                init_method='tcp://{}:{}'.format(
                                    self.server_address[0],
                                    self.server_address[1]),
                                rank=0,
                                world_size=world_size)

class ServerSynchronousTopology(ServerBasicTopology):
    """Synchronous communication class

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
                 server_handler,
                 server_address,
                 dist_backend="gloo",
                 logger=None):
        super(ServerSynchronousTopology,
              self).__init__(handler=server_handler,
                             server_address=server_address,
                             dist_backend=dist_backend)
        
        self._LOGGER = logging if logger is None else logger
        self._LOGGER.info(
            "Server initializes with ip address {}:{} and distributed backend {}"
            .format(server_address[0], server_address[1], dist_backend))

        self.global_round = 3  # for current test
        # TODO 考虑通过KV store实现， 参数同步

    def run(self):
        """Main Process"""
        self._LOGGER.info("Initializing pytorch distributed group\n Waiting for connection requests from clients")
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
            # TODO: give up send_model
            PackageProcessor.send_model(self._handler.model,
                                        MessageCode.ParameterUpdate.value,
                                        dst=client_idx)

    def listen_clients(self):
        """Listen messages from clients"""
        self._handler.train()  # turn train_flag to True
        # server_handler will turn off train_flag once the global model is updated
        while self._handler.train_flag:
            sender, message_code, s_parameters = PackageProcessor.recv_model(
                self._handler.model)

            # 把对message_code的判断信息挪到topology层
            # 因为有的信息不需要下层进行数据操作，而是直接上层返回，handler部分不应该调用network相关模块
            if message_code == MessageCode.ParameterUpdate:
                self._handler.on_receive(sender, message_code, s_parameters)
            else:
                self._LOGGER.info("invalid message code {}".format(message_code))

    def shutdown_clients(self):
        """Shutdown all clients"""
        for client_idx in range(self._handler.client_num_in_total):
            PackageProcessor.send_model(self._handler.model,
                                        MessageCode.Exit.value,
                                        dst=client_idx + 1)

# xiangjing
# 客户端可考虑采用PassiveTopology

class ServerAsynchronousTopology(ServerBasicTopology):
    def __init__(self,
                 server_handler,
                 server_address,
                 dist_backend="gloo",
                 logger=None):

        super(ServerAsynchronousTopology,
              self).__init__(server_handler,
                             server_address=server_address,
                             dist_backend=dist_backend)

        self._LOGGER = logging if logger is None else logger
        self._LOGGER.info(
            "Server initializes with ip address {}:{} and distributed backend {}"
            .format(server_address[0], server_address[1], dist_backend))

        self.global_activate_epoch = 3  # global_round is global epochs in algorithm, for current test
        self.has_new_update = False  # when new update is coming, start the next activate
        self.total_update_time = self.global_activate_epoch * self._handler.client_num_per_round  # to end updater

    def run(self):
        """Main process

        """
        self._LOGGER.info("Initializing pytorch distributed group")
        self._LOGGER.info("Waiting for connection requests from clients")
        self.init_network_connection(
            world_size=self._handler.client_num_in_total + 1)
        self._LOGGER.info("Connect to clients successfully")

        #activate = threading.Thread(target=self.activate_clients)
        listen = threading.Thread(target=self.listen_clients)

        #activate.start()
        listen.start()

        listen.join()
        self.shutdown_clients()

    def activate_clients(self):
        """Activate some of clients to join each FL epoch
           when the updated model is coming, start next FL epoch
        """
        current_model_epoch_torch = torch.zeros(1)
        for current_model_epoch in range(self.global_activate_epoch):
            current_model_epoch_torch[0] = current_model_epoch
            self._LOGGER.info("Global FL epoch {}/{}".format(
                current_model_epoch + 1, self.global_activate_epoch))

            clients_this_epoch = self._handler.select_clients()
            self._LOGGER.info("client id list for this FL round: {}".format(
                clients_this_epoch))

            for client_idx in clients_this_epoch:
                # PackageProcessor.send_model_with_time(
                #     self._handler.model, MessageCode.ParameterUpdate.value, dst=client_idx)
                pack = Package(message_code=MessageCode.ParameterUpdate)
                model_params = SerializationTool.serialize_model(
                    self._handler.model)
                pack.append_tensor_list(
                    [model_params, current_model_epoch_torch])
                PackageProcessor.send_package(pack, dst=client_idx)
            self.has_new_update = False

            while not self.has_new_update:
                pass

    def listen_clients(self):
        """Listen messages from clients"""
        for current_update_time in range(self.total_update_time):
            sender, message_code, content = PackageProcessor.recv_package()

            if message_code == MessageCode.ParameterRequest:
                pass
            elif message_code == MessageCode.ParameterUpdate:
                pass

            self._handler.model_update_time = current_update_time
            self._handler.on_receive(sender, message_code, content)
            self.has_new_update = True

    def shutdown_clients(self):
        """Shutdown all clients"""
        self._LOGGER.info(
            "All updated models from activated clients are received and updated,"
            "totally update {} times".format(self.total_update_time))
        for client_idx in range(self._handler.client_num_in_total):
            model_params = SerializationTool.serialize_model(
                self._handler.model)
            pack = Package(message_code=MessageCode.Exit, content=model_params)
            PackageProcessor.send_package(pack, dst=client_idx + 1)
