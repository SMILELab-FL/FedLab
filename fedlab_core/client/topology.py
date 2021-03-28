from math import log
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import send
from torch.multiprocessing import Process

from fedlab_core.utils.messaging import send_message, recv_message, MessageCode
from fedlab_core.utils.logger import logger


class ClientCommunicationTopology(Process):
    """Abstract class
        if you want to define your own Network Topology
        please be sure your class is derived from this abstract class and OVERRIDE its methods!

        Example:
            please read the code of `ClientSyncTop`
    """
    def __init__(self, backend_handler, server_addr, world_size, rank, dist_backend):
        self._backend = backend_handler

        self.rank = rank
        self.server_addr = server_addr
        self.world_size = world_size
        self.dist_backend = dist_backend

        self._LOGGER = logger("log"+str(rank)+".txt",
                              client_name="client"+str(rank))

        self._LOGGER.info("Initializing --- connecting to server:{},  world_size:{}, rank:{}, backend:{}".format(
            server_addr, world_size, rank, dist_backend))

        dist.init_process_group(backend=dist_backend, init_method='tcp://{}:{}'
                                .format(self.server_addr[0], self.server_addr[1]),
                                rank=self.rank, world_size=self.world_size)

    def run(self):
        # TODO: please override this function
        raise NotImplementedError()

    def receive(self, sender, message_code, payload):
        # TODO: please override this function
        raise NotImplementedError()

    def synchronise(self, payload):
        # TODO: please override this function
        raise NotImplementedError()


class ClientSyncTop(ClientCommunicationTopology):
    """Synchronise conmmunicate class

    This is the top class in our framework which is mainly responsible for network communication of CLIENT!
    Synchronize with server following agreements defined in run().

    Args:
        backend_handler: class derived from ClientBackendHandler
        server_addr: (ip:port) address of server
        world_size: world_size for `torch.distributed` initialization
        rank: rank for `torch.distributed` initialization
        dist_backend: backend of `torch.distributed` (gloo, mpi and ncll) and gloo is default

    Returns:
        None

    Raises:
        Errors raised by `torch.distributed.init_process_group()`
    """
    def __init__(self, backend_handler, server_addr, world_size, rank, dist_backend="gloo"):
        super(ClientSyncTop, self).__init__(backend_handler,
                                            server_addr, world_size, rank, dist_backend)

        self._buff = torch.zeros(
            self._backend.buffer.numel() + 2).cpu()  # 需要修改

    def run(self):
        """Main process of client is defined here:
            1. client waits for data from server
            2. after receiving data, client will train local model
            3. client will synchronize with server actively
        """

        while (True):
            self._LOGGER.info("waiting message from server")
            recv_message(self._buff, src=0)  # 阻塞式
            sender = int(self._buff[0].item())
            message_code = MessageCode(self._buff[1].item())
            parameter = self._buff[2:]

            if message_code == MessageCode.Exit:
                break

            self.receive(sender, message_code, parameter)
            self.synchronise(self._backend.buffer)

    def receive(self, sender, message_code, payload):
        """Synchronise function: reaction of receive new message

        Args:
            sender: index in torch.distributed
            message_code: agreements code defined in MessageCode class
            payload: serialized network parameter (by ravel_model_params function)

        Returns:
            None

        Raises:
            None
        """
        self._LOGGER.info("receiving message from {}, message code {}".format(
            sender, message_code))

        self._backend.buffer = payload
        self._backend.train(epochs=2)

    def synchronise(self, buffer):
        """synchronise local network with server actively

        Args:
            buffer: serialized network parameters

        Returns:
            None

        Raises:
            None
        """
        self._LOGGER.info("synchronise model prameters with server")
        send_message(MessageCode.ParameterUpdate, buffer)
