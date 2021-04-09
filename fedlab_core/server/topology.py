import os
import threading

import torch
import torch.distributed as dist
from torch.multiprocessing import Process, Lock

from fedlab_core.utils.messaging import MessageCode, recv_message, send_message
from fedlab_core.utils.logger import logger

from fedlab_core.client.topology import ClientCommunicationTopology


class EndTop(Process):
    """Abstract class for server network topology

    If you want to define your own topology agreements, please subclass it.

    Args:
        server_handler: Parameter server backend handler derived from :class:`ParameterServerHandler`
        server_address (tuple): Address of server in form of ``(SERVER_ADDR, SERVER_IP)``
        dist_backend (str or Backend): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
        and ``nccl``
    """

    def __init__(self, server_handler, server_address, dist_backend):
        self._handler = server_handler
        self.server_address = server_address  # ip:port
        self.dist_backend = dist_backend

    def run(self):
        """Process"""
        raise NotImplementedError()

    def activate_clients(self):
        """Activate some of clients to join this FL round"""
        raise NotImplementedError()

    def listen_clients(self):
        """Listen messages from clients"""
        raise NotImplementedError()


class ServerSyncTop(EndTop):
    """Synchronous communication class

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronize with clients following agreements defined in :meth:`run`.

    Args:
        server_handler: Subclass of :class:`ParameterServerHandler`
        server_address (tuple): Address of this server in form of ``(SERVER_ADDR, SERVER_IP)``
        dist_backend (str or Backend): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
        and ``nccl``. Default: ``"gloo"``
        logger_path (str, optional): path to the log file for this class. Default: ``"server_top.txt"``
        logger_name (str, optional): class name to initialize logger. Default: ``"ServerTop"``

    Raises:
        None
    """

    def __init__(self, server_handler, server_address, dist_backend="gloo", logger_path="server_top.txt",
                 logger_name="ServerTop"):

        super(ServerSyncTop, self).__init__(server_handler=server_handler,
                                            server_address=server_address, dist_backend=dist_backend)
        self.buff = torch.zeros(
            self._handler.buffer.numel() + 2).cpu()  # TODO: 通信信息缓存 模型参数+2个控制参数位，need to be more formal

        self._LOGGER = logger(logger_path, logger_name)
        self._LOGGER.info("Server initializes with ip address {}:{} and distributed backend {}".format(
            server_address[0], server_address[1], dist_backend))

    def run(self):
        """Process"""
        self._LOGGER.info("Initializing pytorch distributed group")
        self._LOGGER.info("Waiting for connection requests from clients")
        dist.init_process_group(backend=self.dist_backend, init_method='tcp://{}:{}'
                                .format(self.server_address[0], self.server_address[1]),
                                rank=0, world_size=self._handler.client_num_in_total + 1)
        self._LOGGER.info("Connect to clients successfully")

        global_epoch = 3  # test TODO
        for i in range(global_epoch):
            self._LOGGER.info(
                "Global FL round {}/{}".format(i, global_epoch))

            activate = threading.Thread(target=self.activate_clients)
            listen = threading.Thread(target=self.listen_clients)

            activate.start()
            listen.start()

            activate.join()
            listen.join()

        for index in range(self._handler.client_num):
            end_message = torch.Tensor([0])
            send_message(MessageCode.Exit, payload=end_message, dst=index)

    def activate_clients(self):
        """activate some of clients to join this FL round"""
        usr_list = self._handler.select_clients()
        payload = self._handler.buffer

        self._LOGGER.info(
            "client id list for this FL round: {}".format(usr_list))
        for index in usr_list:
            send_message(MessageCode.ParameterUpdate, payload, dst=index)

    def listen_clients(self):
        """listen messages from clients"""
        self._handler.train()  # flip the update_flag

        # server_handler will turn this flag to True when model parameters updated
        while self._handler.update_flag:
            recv_message(self.buff)
            sender = int(self.buff[0].item())
            message_code = MessageCode(self.buff[1].item())
            parameter = self.buff[2:]

            self._handler.receive(sender, message_code, parameter)
