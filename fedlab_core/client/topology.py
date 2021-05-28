import os
from abc import ABC, abstractmethod

import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab_utils.logger import logger
from fedlab_utils.message_code import MessageCode
from fedlab_core.communicator.processor import PackageProcessor


class ClientBasicTopology(Process, ABC):
    """Abstract class

    If you want to define your own Network Topology, please be sure your class should subclass it and OVERRIDE its
    methods.

    Example:
        Read the code of :class:`ClientSyncTop` to learn how to use this class.
    """

    def __init__(self, server_addr, world_size, rank, dist_backend):
        self.rank = rank
        self.server_addr = server_addr
        self.world_size = world_size
        self.dist_backend = dist_backend

    @abstractmethod
    def run(self):
        """Please override this function"""
        raise NotImplementedError()

    @abstractmethod
    def on_receive(self, sender_rank, message_code, payload):
        """Please override this function"""
        raise NotImplementedError()

    @abstractmethod
    def synchronize(self):
        """Please override this function"""
        raise NotImplementedError()

    def init_network_connection(self):
        dist.init_process_group(backend=self.dist_backend, init_method='tcp://{}:{}'
                                .format(self.server_addr[0], self.server_addr[1]),
                                rank=self.rank, world_size=self.world_size)


class ClientSyncTop(ClientBasicTopology):
    """Synchronize communication class

    This is the top class in our framework which is mainly responsible for network communication of CLIENT!
    Synchronize with server following agreements defined in :meth:`run`.

    Args:
        client_handler: Subclass of ClientBackendHandler, manages training and evaluation of local model on each
        client.
        server_addr (tuple): Address of server in form of ``(SERVER_ADDR, SERVER_IP)``
        world_size (int): Number of client processes participating in the job for ``torch.distributed`` initialization
        rank (int): Rank of the current client process for ``torch.distributed`` initialization
        dist_backend (str or Backend): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
        and ``nccl``. Default: ``"gloo"``
        logger_file (str, optional): Path to the log file for all clients of :class:`ClientSyncTop` class. Default: ``"client_log"``
        logger_name (str, optional): Class name to initialize logger. Default: ``""``

    Raises:
        Errors raised by :func:`torch.distributed.init_process_group`
    """

    def __init__(self, client_handler, server_addr, world_size, rank, dist_backend="gloo",
                 logger_file="client_log",
                 logger_name="client"):

        super(ClientSyncTop, self).__init__(
            server_addr, world_size, rank, dist_backend)

        self._handler = client_handler

        self.epochs = 2  # epochs for local training

        self._LOGGER = logger(os.path.join(
            "log", logger_file + str(rank) + ".txt"), logger_name+str(rank))

    def run(self):
        """Main procedure of each client is defined here:
            1. client waits for data from server
            2. after receiving data, client will train local model
            3. client will synchronize with server actively
        """
        self._LOGGER.info("connecting with server")
        self.init_network_connection()
        self._LOGGER.info(
            "connected to server:{}:{},  world size:{}, rank:{}, backend:{}".format(
                self.server_addr[0], self.server_addr[1], self.world_size, self.rank, self.dist_backend))
        while True:
            self._LOGGER.info("Waiting for server...")
            # waits for data from
            sender_rank, message_code, s_parameters = PackageProcessor.recv_model(
                self._handler.model, src=0)

            # exit
            if message_code == MessageCode.Exit:
                self._LOGGER.info(
                    "Recv {}, Process exiting".format(message_code))
                exit(0)

            # perform local training
            self.on_receive(sender_rank, message_code, s_parameters)

            # synchronize with server
            self.synchronize()

    def on_receive(self, sender_rank, message_code, s_parameters):
        """Actions to perform on receiving new message, including local training

        Args:
            sender_rank (int): Rank of sender
            message_code (MessageCode): Agreements code defined in: class:`MessageCode`
            s_parameters (torch.Tensor): Serialized model parameters
        """
        self._LOGGER.info("Package received from {}, message code {}".format(
            sender_rank, message_code))

        self._handler.load_parameters(s_parameters)
        self._handler.train(epochs=self.epochs)

    def synchronize(self):
        """Synchronize local model with server actively"""
        self._LOGGER.info("synchronize model parameters with server")
        PackageProcessor.send_model(
            self._handler.model, MessageCode.ParameterUpdate.value, dst=0)
