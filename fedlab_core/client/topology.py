import os

import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab_core.utils.logger import logger
from fedlab_core.message_processor import MessageProcessor
from fedlab_core.utils.message_code import MessageCode


# list or tensor, for this example header = [message code]
HEADER_INSTANCE = [1]


class ClientBasicTop(Process):
    """Abstract class

    If you want to define your own Network Topology, please be sure your class should subclass it and OVERRIDE its
    methods.

    Example:
        please read the code of :class:`ClientSyncTop`
    """

    def __init__(self, server_addr, world_size, rank, dist_backend):
        self.rank = rank
        self.server_addr = server_addr
        self.world_size = world_size
        self.dist_backend = dist_backend

    def run(self):
        """Please override this function"""
        raise NotImplementedError()

    def on_receive(self, sender, message_code, payload):
        """Please override this function"""
        raise NotImplementedError()

    def synchronize(self):
        """Please override this function"""
        raise NotImplementedError()

    def init_network_connection(self):
        dist.init_process_group(backend=self.dist_backend, init_method='tcp://{}:{}'
                                .format(self.server_addr[0], self.server_addr[1]),
                                rank=self.rank, world_size=self.world_size)


class ClientSyncTop(ClientBasicTop):
    """Synchronise communication class

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
        logger_file (str, optional): Path to the log file for all clients of :class:`ClientSyncTop` class. Default: ``"clientLog"``
        logger_name (str, optional): Class name to initialize logger

    Raises:
        Errors raised by :func:`torch.distributed.init_process_group`
    """

    def __init__(self, client_handler, server_addr, world_size, rank, dist_backend="gloo", logger_file="clientLog",
                 logger_name=""):

        super(ClientSyncTop, self).__init__(
            server_addr, world_size, rank, dist_backend)

        self._backend = client_handler

        self._LOGGER = logger(os.path.join(
            "log", logger_file + str(rank) + ".txt"), logger_name)
        self._LOGGER.info(
            "Successfully Initialized --- connected to server:{}:{},  world size:{}, rank:{}, backend:{}".format(
                server_addr[0], server_addr[1], world_size, rank, dist_backend))

        self.msg_processor = MessageProcessor(
            header_instance=HEADER_INSTANCE, model=self._backend.model)

    def run(self):
        """Main procedure of each client is defined here:
            1. client waits for data from server
            2. after receiving data, client will train local model
            3. client will synchronize with server actively
        """
        self._LOGGER.info("connecting with server")
        self.init_network_connection()
        while True:
            # waits for data from
            package = self.msg_processor.recv_package(src=0)
            sender, _, message_code, s_parameters = self.msg_processor.unpack(
                payload=package)

            # exit
            if message_code == MessageCode.Exit:
                exit(0)

            # perform local training
            self.on_receive(sender, message_code, s_parameters)

            # synchronize with server
            self.synchronize()

    def on_receive(self, sender, message_code, s_parameters):
        """Actions to perform on receiving new message, including local training

        Args:
            sender (int): Rank of sender
            message_code (MessageCode): Agreements code defined in: class:`MessageCode`
            s_parameters (torch.Tensor): Serialized model parameters
        """
        self._LOGGER.info("receiving message from {}, message code {}".format(
            sender, message_code))

        self._backend.load_parameters(s_parameters)
        self._backend.train(epochs=2)

    def synchronize(self):
        """Synchronize local model with server actively"""
        self._LOGGER.info("synchronize model parameters with server")
        payload = self.msg_processor.pack(
            header=[MessageCode.ParameterUpdate.value], model=self._backend.model)
        self.msg_processor.send_package(payload=payload, dst=0)
