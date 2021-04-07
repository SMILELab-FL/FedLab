import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab_core.utils.messaging import MessageCode, recv_message, send_message

from fedlab_core.client.topology import ClientCommunicationTopology
from fedlab_core.server.topology import EndTop


class PipeTop(Process):
    """
    Abstract class for server Pipe topology
    simple example
    """

    def __init__(self, model, server_process, client_process):

        self._model = model

        self.sp = server_process
        self.cp = client_process

        # 临界资源
        # 子进程同步

    def run(self):
        """process function"""
        
        self.sp.start()
        self.cp.start()

        self.sp.join()
        self.cp.join()


class ConnectClient(EndTop):
    """Provide service to clients as a middle server"""

    def __init__(self, handler, server_address, world_size, dist_backend='gloo'):

        super(ConnectClient, self).__init__(
            handler, server_address, dist_backend)

        #self.locks = locks

    def run(self):
        """ """
        dist.init_process_group(backend=self.dist_backend, init_method='tcp://{}:{}'
                                .format(self.server_address[0], self.server_address[1]),
                                rank=0, world_size=self._handler.client_num+1)

    def activate_clients(self):
        """activate some of clients to join this FL round"""
        usr_list = self._handler.select_clients()
        payload = self._handler.buffer

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
    
    def share_buffer(self, buffer, lock):
        raise NotImplementedError()


class ConnectServer(ClientCommunicationTopology):
    """connect to upper server"""

    def __init__(self, locks, server_address, world_size, rank, dist_backend='gloo'):
        # connect server
        dist.init_process_group(backend=dist_backend, init_method='tcp://{}:{}'
                                .format(server_address[0], server_address[1]),
                                rank=rank, world_size=world_size)

    def run(self):
        raise NotImplementedError()

    def receive(self, sender, message_code, payload):
        return super().receive(sender, message_code, payload)

    def synchronise(self, payload):
        return super().synchronise(payload)

    def share_buffer(self, buffer, lock):
        raise NotImplementedError()
