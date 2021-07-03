import logging
from abc import ABC, abstractmethod

import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab_core.topology import Topology
from fedlab_utils.message_code import MessageCode
from fedlab_utils.serialization import SerializationTool
from fedlab_core.communicator.processor import Package, PackageProcessor


class ClientBasicTopology(Process, ABC):
    """Abstract class of client topology

    If you want to define your own Network Topology, please be sure your class should subclass it and OVERRIDE its methods.

    Example:
        Read the code of :class:`ClientPassiveTopology` and `ClientActiveTopology` to learn how to use this class.
    """
    def __init__(self, handler, network):
        self._handler = handler
        self._network = network

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


class ClientPassiveTopology(Topology):
    """Passive communication topology

    Args:
        client_handler: Subclass of ClientBackendHandler. Provides meth:train and attribute:model
        server_addr (tuple): Address of server in form of ``(SERVER_ADDR, SERVER_IP)``
        world_size (int): Number of client processes participating in the job for ``torch.distributed`` initialization rank (int): Rank of the current client process for ``torch.distributed`` initialization
        dist_backend (str or Backend): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``, and ``nccl``. Default: ``"gloo"``
        logger (`logger`, optional): object of `fedlab_utils.logger`
    """

    def __init__(self, handler, network, logger=None):
        super().__init__(handler, network)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def run(self):
        """Main procedure of each client is defined here:
            1. client waits for data from server （PASSIVE）
            2. after receiving data, client will train local model
            3. client will synchronize with server actively
        """
        self._LOGGER.info("connecting with server")
        self._network.init_network_connection()
        while True:
            self._LOGGER.info("Waiting for server...")
            # waits for data from
            sender_rank, message_code, payload = PackageProcessor.recv_package(
                src=0)
            # exit
            if message_code == MessageCode.Exit:
                self._LOGGER.info(
                    "Recv {}, Process exiting".format(message_code))
                exit(0)
            else:
                # perform local training
                self.on_receive(sender_rank, message_code, payload)

            # synchronize with server
            self.synchronize()

    def on_receive(self, sender_rank, message_code, payload):
        """Actions to perform on receiving new message, including local training

        Args:
            sender_rank (int): Rank of sender
            message_code (MessageCode): Agreements code defined in: class:`MessageCode`
            payload (torch.Tensor): Serialized parameters
        """
        self._LOGGER.info("Package received from {}, message code {}".format(
            sender_rank, message_code))
        s_parameters = payload[0]
        self._handler.train(model_parameters=s_parameters)

    def synchronize(self):
        """Synchronize local model with server actively"""
        self._LOGGER.info("synchronize model parameters with server")
        model_params = SerializationTool.serialize_model(self._handler.model)
        pack = Package(message_code=MessageCode.ParameterUpdate,
                       content=model_params)
        PackageProcessor.send_package(pack, dst=0)


class ClientActiveTopology(Topology):
    """Active communication topology

        Args:
            client_handler: Subclass of ClientBackendHandler, manages training and evaluation of local model on each
            client.
            server_addr (tuple): Address of server in form of ``(SERVER_ADDR, SERVER_IP)``
            world_size (int): Number of client processes participating in the job for ``torch.distributed`` initialization
            rank (int): Rank of the current client process for ``torch.distributed`` initialization
            dist_backend (str or Backend): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
            and ``nccl``. Default: ``"gloo"``
            epochs (int): epochs for local train
            logger (`logger`, optional): object of `fedlab_utils.logger`
    """
    def __init__(self, handler, network, local_epochs=None, logger=None):
        super().__init__(handler, network)

        # temp variables, can assign train epoch rather than initial epoch value in handler
        self.epochs = local_epochs
        self.model_gen_time = None  # record received model's generated update time

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def run(self):
        """Main procedure of each client is defined here:
            1. client requests data from server (ACTIVE)
            2. after receiving data, client will train local model
            3. client will synchronize with server actively
        """
        self._LOGGER.info("connecting with server")
        self._network.init_network_connection()
        while True:
            self._LOGGER.info("Waiting for server...")
            # request model actively
            self.request_model()
            # waits for data from
            sender_rank, message_code, payload = PackageProcessor.recv_package(
                src=0)

            # exit
            if message_code == MessageCode.Exit:
                self._LOGGER.info(
                    "Recv {}, Process exiting".format(message_code))
                exit(0)

            # perform local training
            self.on_receive(sender_rank, message_code, payload)

            # synchronize with server
            self.synchronize()

    def on_receive(self, sender_rank, message_code, payload):
        """Actions to perform on receiving new message, including local training

        Args:
            sender_rank (int): Rank of sender
            message_code (MessageCode): Agreements code defined in: class:`MessageCode`
            s_parameters (torch.Tensor): Serialized model parameters
        """
        self._LOGGER.info("Package received from {}, message code {}".format(
            sender_rank, message_code))
        s_parameters = payload[0]
        self.model_gen_time = payload[1]
        # move loading model params to the start of training
        self._handler.train(epochs=self.epochs, model_parameters=s_parameters)

    def synchronize(self):
        """Synchronize local model with server actively"""
        self._LOGGER.info("synchronize model parameters with server")
        model_params = SerializationTool.serialize_model(self._handler.model)
        pack = Package(message_code=MessageCode.ParameterUpdate)
        pack.append_tensor_list([model_params, self.model_gen_time])
        PackageProcessor.send_package(pack, dst=0)

    def request_model(self):
        """send ParameterRequest"""
        self._LOGGER.info("request model parameters from server")
        pack = Package(message_code=MessageCode.ParameterRequest)
        PackageProcessor.send_package(pack, dst=0)
