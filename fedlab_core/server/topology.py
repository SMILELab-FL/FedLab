import os
import threading

import torch
import torch.distributed as dist
from torch.multiprocessing import Process, Lock

from fedlab_core.utils.messaging import MessageCode, recv_message, send_message


class EndTop(Process):
    """Synchronise communicate Class

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronize with clients following agreements defined in run().

    Args:
        ParameterServerHandler: a class derived from ParameterServerHandler
        args: params

    Returns:
        None
    Raises:
        None

    """

    def __init__(self, ParameterServerHandler, server_addr, logger, dist_backend="gloo"):

        self._params_server = ParameterServerHandler

        self.server_addr = server_addr
        self.dist_backend = dist_backend

        self.buff = torch.zeros(
            self._params_server.buffer.numel() + 2).cpu()  # 通信信息缓存 模型参数+2

        self._LOGGER = logger

        self._LOGGER.info("Server initailize with ip address {} and distributed backend {}".format(
            server_addr, dist_backend))

    def run(self):
        """Process function"""
        self._LOGGER.info(
            "Waiting for the connection request from clients!")
        dist.init_process_group(backend=self.dist_backend, init_method='tcp://{}:{}'
                                .format(self.server_addr[0], self.server_addr[1]),
                                rank=0, world_size=self._params_server.client_num + 1)
        self._LOGGER.info("Connect to client successfully!")

        act_clients = threading.Thread(target=self.activate_clients)
        wait_info = threading.Thread(target=self.listen_clients)

        self.running = 3
        for i in range(self.running):
            self._LOGGER.info(
                "Global FL round {}/{}".format(i, self.running))

            # 开启选取参与者线程
            act_clients.start()
            # 开启接收回信线程
            wait_info.start()

            # join 方法阻塞主线程
            act_clients.join()
            wait_info.join()

        for index in range(self._params_server.client_num):
            send_message(MessageCode.Exit, payload=None, dst=index)

    def activate_clients(self):
        """activate some of clients to join this FL round"""
        usr_list = self._params_server.select_clients()
        payload = self._params_server.buffer

        self._LOGGER.info(
            "client id list for this FL round: {}".format(usr_list))
        for index in usr_list:
            send_message(MessageCode.ParameterUpdate, payload, dst=index)

    def listen_clients(self):
        """listen messages from clients"""
        self._params_server.start_round()
        while (True):
            recv_message(self.buff)
            sender = int(self.buff[0].item())
            message_code = MessageCode(self.buff[1].item())
            parameter = self.buff[2:]

            self._params_server.receive(sender, message_code, parameter)

            if self._params_server.is_updated():
                break

# PipeTop(EndTop)
class PipeTop(Process):
    """
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

        self._model = model

        locks = []
        [locks.append(Lock()) for _ in range(client_num)]

        self.upper_p = ConnectServer(
            locks, server_dist_info["address"], server_dist_info["world_size"], server_dist_info["rank"], server_dist_info["backend"])

        self.lower_p = ConnectClient(
            locks, client_dist_info["address"], server_dist_info["world_size"], server_dist_info["rank"], server_dist_info["backend"])

    def run(self):
        """process function"""
        self.upper_p.start()
        self.lower_p.start()

        self.upper_p.join()  # 阻塞主进程
        self.lower_p.join()


# considering derived from EndTop
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
