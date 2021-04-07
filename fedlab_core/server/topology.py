import os
import threading

import torch
import torch.distributed as dist
from torch.multiprocessing import Process, Lock

from fedlab_core.utils.messaging import MessageCode, recv_message, send_message
from fedlab_core.utils.logger import logger

from fedlab_core.client.topology import ClientCommunicationTopology


class EndTop(Process):
    """
    Abstract class for server network topology

    If you want to define your own topology agreements, please overide this class.

    args:
        handler: back-end of server
        server_address: (ip,port) to init `torch`
        dist_backend: parameter to init `torch.distributed`
    """

    def __init__(self, handler, server_address, dist_backend):
        self._handler = handler
        self.server_address = server_address  # ip:port
        self.dist_backend = dist_backend

    def run(self):
        """Process"""
        raise NotImplementedError()

    def activate_clients(self):
        """activate some of clients to join this FL round"""
        raise NotImplementedError()

    def listen_clients(self):
        """listen messages from clients"""
        raise NotImplementedError()


class ServerSyncTop(EndTop):
    """Synchronise communicate Class

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronize with clients following agreements defined in run().

    Args:
        ParameterServerHandler: a class derived from ParameterServerHandler
        server_address: (ip:port) ipadress for `torch.distributed` initialization, because this is a server, rank is set by 0.
        dist_backend: backend of `torch.distributed` (gloo, mpi and ncll) and gloo is default
        logger_file: path to the log file for this class
        logger_name: class name to initialize logger

    Raises:
        None

    """

    def __init__(self, ParameterServerHandler, server_address, dist_backend="gloo", logger_path="server_top.txt", logger_name="ServerTop"):

        super(ServerSyncTop, self).__init__(handler=ParameterServerHandler,
                                            server_address=server_address, dist_backend=dist_backend)
        self.buff = torch.zeros(
            self._handler.buffer.numel() + 2).cpu()  # 通信信息缓存 模型参数+2个控制参数位

        self._LOGGER = logger(logger_path, logger_name)
        self._LOGGER.info("Server initailize with ip address {} and distributed backend {}".format(
            server_address, dist_backend))

    def run(self):
        """Process"""
        self._LOGGER.info("Initializing pytorch distributed group")
        self._LOGGER.info("Waiting for the connection request from clients")
        dist.init_process_group(backend=self.dist_backend, init_method='tcp://{}:{}'
                                .format(self.server_address[0], self.server_address[1]),
                                rank=0, world_size=self._handler.client_num + 1)
        self._LOGGER.info("Connect to client successfully")

        global_epoch = 3    # test TODO 
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
        self._handler.start_round()  # flip the update_flag
        while (True):
            recv_message(self.buff)
            sender = int(self.buff[0].item())
            message_code = MessageCode(self.buff[1].item())
            parameter = self.buff[2:]

            self._handler.receive(sender, message_code, parameter)

            if self._handler.update_flag:
                # server_handler will turn this flag to True when model parameters updated
                self._LOGGER.info("updated quit listen")
                break
