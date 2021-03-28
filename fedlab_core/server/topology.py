import os
import threading

import torch
import torch.distributed as dist
from torch.multiprocessing import Process, Lock

from fedlab_core.utils.messaging import MessageCode, recv_message, send_message
from fedlab_core.utils.logger import logger


class EndTop(Process):
    """Synchronise communicate Class

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronize with clients following agreements defined in run().

    Args:
        ParameterServerHandler: a class derived from ParameterServerHandler
        server_addr: (ip:port) ipadress for `torch.distributed` initialization, because this is a server, rank is set by 0.
        dist_backend: backend of `torch.distributed` (gloo, mpi and ncll) and gloo is default
        logger_file: path to the log file for this class
        logger_name: class name to initialize logger

    Raises:
        None

    """

    def __init__(self, ParameterServerHandler, server_addr, dist_backend="gloo", logger_path="server_top.txt", logger_name="ServerTop"):

        self._server_handler = ParameterServerHandler

        self.server_addr = server_addr
        self.dist_backend = dist_backend

        self.buff = torch.zeros(
            self._server_handler.buffer.numel() + 2).cpu()  # 通信信息缓存 模型参数+2

        self._LOGGER = logger(logger_path, logger_name)
        self._LOGGER.info("Server initailize with ip address {} and distributed backend {}".format(
            server_addr, dist_backend))

    def run(self):
        """Process"""
        self._LOGGER.info(
            "Waiting for the connection request from clients")
        dist.init_process_group(backend=self.dist_backend, init_method='tcp://{}:{}'
                                .format(self.server_addr[0], self.server_addr[1]),
                                rank=0, world_size=self._server_handler.client_num + 1)
        self._LOGGER.info("Connect to client successfully")

        self.running = 3
        for i in range(self.running):
            self._LOGGER.info(
                "Global FL round {}/{}".format(i, self.running))
            act_clients = threading.Thread(target=self.activate_clients)
            wait_info = threading.Thread(target=self.listen_clients)

            act_clients.start()
            wait_info.start()

            act_clients.join()
            wait_info.join()

        for index in range(self._server_handler.client_num):
            end_message = torch.Tensor([0])
            send_message(MessageCode.Exit, payload=end_message, dst=index)

    def activate_clients(self):
        """activate some of clients to join this FL round"""
        usr_list = self._server_handler.select_clients()
        payload = self._server_handler.buffer

        self._LOGGER.info(
            "client id list for this FL round: {}".format(usr_list))
        for index in usr_list:
            send_message(MessageCode.ParameterUpdate, payload, dst=index)

    def listen_clients(self):
        """listen messages from clients"""
        self._server_handler.start_round()  # flip the update_flag
        while (True):
            recv_message(self.buff)
            sender = int(self.buff[0].item())
            message_code = MessageCode(self.buff[1].item())
            parameter = self.buff[2:]

            self._server_handler.receive(sender, message_code, parameter)

            if self._server_handler.update_flag:
                # server_handler will turn this flag to True when model parameters updated
                self._LOGGER("updated quit listen")
                break


class PipeTop(EndTop):
    """
    TODO
    the frame of PipeTop is not settled!!!
    This process help implement hierarchical parameter server
    it connects clients and Top Server directly, collecting information from clients and sending to Top Server

    Args:
        model: TODO

        client_num:  TODO

        server_dist_info:

        client_dist_info:

    Returns:
        None
    Raises:
        None

    """

    def __init__(self, model, client_num, server_dist_info, client_dist_info):
        raise NotImplementedError()

    def run(self):
        """process function"""
        raise NotImplementedError()


class ConnectClient(Process):
    """Provide service to clients as a middle server"""

    def __init__(self, locks, local_addr, world_size, rank=0, dist_backend='gloo'):

        self.locks = locks

        self.dist_backend = dist_backend
        self.local_addr = local_addr
        self.word_size = world_size
        self.rank = rank
        self.dist_backend = dist_backend

    def run(self):
        """ """
        dist.init_process_group(backend=self.dist_backend, init_method='tcp://{}:{}'
                                .format(self.local_addr[0], self.local_addr[1]),
                                rank=self.rank, world_size=self.world_size)


class ConnectServer(Process):
    """connect to upper server"""

    def __init__(self, locks, server_addr, world_size, rank, dist_backend='gloo'):

        # connect server
        dist.init_process_group(backend=dist_backend, init_method='tcp://{}:{}'
                                .format(server_addr[0], server_addr[1]),
                                rank=rank, world_size=world_size)

    def run(self):
        raise NotImplementedError()
