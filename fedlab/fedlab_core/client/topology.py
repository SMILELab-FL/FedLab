
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab_core.utils.messaging import send_message, recv_message, MessageCode


class ClientCommunicationTopology(Process):
    """Abstract class"""
    def __init__(self, backend_handler) -> None:
        super().__init__()
        self._backend = backend_handler

    def run(self):
        pass

    def receive(self, sender, message_code, payload):
        pass

    def synchronise(self, payload):
        pass


class ClientSyncTop(ClientCommunicationTopology):
    """Synchronise conmmunicate Class
       This is the top class in our framework which is mainly responsible for network communication of CLIENT!
       Synchronize with server following agreements defined in run().
    """
    def __init__(self, backend_handler, server_addr, world_size, rank, dist_backend="gloo", args=None):
        """ Constructor 
        Args:
            backend_handler: class derived from ClientBackendHandler
            server_addr: (ip:port) address of server
            world_size: world_size for torch.distributed initialization
            rank: rank for torch.distributed initialization
            args: other params
        Returns:
            None
        Raises:
            Errors raised by torch.distributed.init_process_group()
        """
        self._backend = backend_handler

        # distributed init params
        self.rank = rank
        self.server_addr = server_addr
        self.word_size = world_size
        self.dist_backend = dist_backend

        self._buff = torch.zeros(
            self._backend.get_buff().numel() + 2).cpu()  # 需要修改
        self.args = args

        dist.init_process_group(backend=dist_backend, init_method='tcp://{}:{}'
                                .format(self.server_addr[0], self.server_addr[1]),
                                rank=self.rank, world_size=self.world_size)

        super().__init__()

    def run(self):
        """Main process of client is defined here"""
        while(True):
            print("waiting message from server...")
            recv_message(self._buff, src=0)  # 阻塞式
            sender = int(self._buff[0].item())
            message_code = MessageCode(self._buff[1].item())
            parameter = self._buff[2:]

            # need logger
            self.receive(sender, message_code, parameter)
            self.synchronise(self._backend.get_buff())
            print("synchronized...")

            # 虚添加中止该进程的方法

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
        self._backend.update_model(payload)
        self._backend.train(self.args)

    def synchronise(self, buffer):
        """synchronise local network with server

        Args:
            buffer: serialized network parameters

        Returns:
            None

        Raises:
            None
        """
        send_message(MessageCode.ParameterUpdate, buffer)
