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
    """

    def __init__(self, handler, server_address, dist_backend):
        self._server_handler = handler
        self.server_addr = server_address  # ip:port
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


class PipeTop(Process):
    """
    Abstract class for server Pipe topology
    """

    def __init__(self, model, client_num, server_dist_info, client_dist_info):

        self.connect_clients_process = None
        self.connect_server_process = None

        # 临界资源
        # 子进程同步

    def run(self):
        """process function"""

        self.connect_clients_process.start()
        self.connect_server_process.start()

        self.connect_clients_process.join()
        self.connect_server_process.join()


class ConnectClient(EndTop):
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
    
    def activate_clients(self):
        return super().activate_clients()

    def listen_clients(self):
        return super().listen_clients()


class ConnectServer(ClientCommunicationTopology):
    """connect to upper server"""

    def __init__(self, locks, server_addr, world_size, rank, dist_backend='gloo'):
        # connect server
        dist.init_process_group(backend=dist_backend, init_method='tcp://{}:{}'
                                .format(server_addr[0], server_addr[1]),
                                rank=rank, world_size=world_size)

    def run(self):
        raise NotImplementedError()

    def receive(self, sender, message_code, payload):
        return super().receive(sender, message_code, payload)

    def synchronise(self, payload):
        return super().synchronise(payload)


class ServerSyncTop(EndTop):
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

    def __init__(self, ParameterServerHandler, server_address, dist_backend="gloo", logger_path="server_top.txt", logger_name="ServerTop"):

        super(ServerSyncTop, self).__init__(handler=ParameterServerHandler,
                                            server_address=server_address, dist_backend=dist_backend)
        self.buff = torch.zeros(
            self._server_handler.buffer.numel() + 2).cpu()  # 通信信息缓存 模型参数+2个控制参数位

        self._LOGGER = logger(logger_path, logger_name)
        self._LOGGER.info("Server initailize with ip address {} and distributed backend {}".format(
            server_address, dist_backend))

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

            # this part could be better
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
                self._LOGGER.info("updated quit listen")
                break
