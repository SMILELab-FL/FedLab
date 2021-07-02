import logging
from abc import ABC, abstractmethod

import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab_utils.message_code import MessageCode
from fedlab_utils.serialization import SerializationTool
from fedlab_core.communicator.processor import Package, PackageProcessor


class ClientBasicTopology(Process, ABC):
    """Abstract class of client topology

    If you want to define your own Network Topology, please be sure your class should subclass it and OVERRIDE its methods.

    Example:
        Read the code of :class:`ClientPassiveTopology` and `ClientActiveTopology` to learn how to use this class.
    """
    def __init__(self, handler, server_addr, world_size, rank, dist_backend):

        self._handler = handler
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
        dist.init_process_group(backend=self.dist_backend,
                                init_method='tcp://{}:{}'.format(
                                    self.server_addr[0], self.server_addr[1]),
                                rank=self.rank,
                                world_size=self.world_size)


"""
client的架构不应该被分为同步和异步，而是应该按照被调用算力的方式分为
    主动网络拓扑： 完成计算就上传并开启下一轮训练
    被动网络拓扑： 等待上层网络调用，才开始训练
根据上述两种分类，添加两个新的架构类ClientActiveTopology、ClientPassiveTopology
原有的同步和异步类被弃用
"""
class ClientPassiveTopology(ClientBasicTopology):
    """Passive communication topology

    Args:
        client_handler: Subclass of ClientBackendHandler. Provides meth:train and attribute:model
        server_addr (tuple): Address of server in form of ``(SERVER_ADDR, SERVER_IP)``
        world_size (int): Number of client processes participating in the job for ``torch.distributed`` initialization
        rank (int): Rank of the current client process for ``torch.distributed`` initialization
        dist_backend (str or Backend): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
        and ``nccl``. Default: ``"gloo"``
        epochs (int): epochs for local train
        logger (`logger`, optional): object of `fedlab_utils.logger`

    """
    def __init__(self,
                 handler,
                 server_addr,
                 world_size,
                 rank,
                 dist_backend='gloo',
                 logger=None):
        super().__init__(handler, server_addr, world_size, rank, dist_backend)

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
        self._LOGGER.info(
            "connected to server:{}:{},  world size:{}, rank:{}, backend:{}".
            format(self.server_addr[0], self.server_addr[1], self.world_size,
                   self.rank, self.dist_backend))

        self.init_network_connection()
        while True:
            self._LOGGER.info("Waiting for server...")
            # waits for data from
            sender_rank, message_code, payload = PackageProcessor.recv_package(src=0)
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
            s_parameters (torch.Tensor): Serialized model parameters
        """
        self._LOGGER.info("Package received from {}, message code {}".format(
            sender_rank, message_code))
        s_parameters = payload[0]
        #self._handler.load_parameters(s_parameters) # put restoring model in train() before training
        self._handler.train(model_parameters=s_parameters)

    def synchronize(self):
        """Synchronize local model with server actively"""
        self._LOGGER.info("synchronize model parameters with server")
        model_params = SerializationTool.serialize_model(self._handler.model)
        pack = Package(message_code=MessageCode.ParameterUpdate, content=model_params)
        PackageProcessor.send_package(pack, dst=0)

        """
        PackageProcessor.send_model(self._handler.model,
                                MessageCode.ParameterUpdate.value,
                                    dst=0)
        """

class ClientActiveTopology(ClientBasicTopology):
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
    def __init__(self,
                 handler,
                 server_addr,
                 world_size,
                 rank,
                 local_epochs=None,
                 dist_backend='gloo',
                 logger=None):
        super().__init__(handler, server_addr, world_size, rank, dist_backend)
        
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
        self._LOGGER.info(
            "connected to server:{}:{},  world size:{}, rank:{}, backend:{}".
            format(self.server_addr[0], self.server_addr[1], self.world_size,
                   self.rank, self.dist_backend))

        self.init_network_connection()
        while True:
            self._LOGGER.info("Waiting for server...")
            # request model actively
            self.request_model()
            # waits for data from
            sender_rank, message_code, payload = PackageProcessor.recv_package(src=0)

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
        self._LOGGER.info("request model parameters from server")
        pack = Package(message_code=MessageCode.ParameterRequest)
        PackageProcessor.send_package(pack, dst=0)
