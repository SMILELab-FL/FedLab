
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab_core.utils.messaging import send_message, recv_message, MessageCode


class ClientCommunicationTopology(Process):

    def __init__(self, backend_handler) -> None:
        super().__init__()
        self._backend = backend_handler

    def run(self):
        pass

    def receive(self, sender, message_code, payload):
        pass

    def synchronise(self, payload):
        pass


class ClientSyncTop(Process):
    def __init__(self, worker, args):
        self._worker = worker
        self._buff = torch.zeros(worker.get_buff().numel() + 2).cpu() # 需要修改
        self.args = args
        dist.init_process_group(backend="gloo", init_method='tcp://{}:{}'
                                .format(args.server_ip, args.server_port),
                                rank=args.local_rank, world_size=args.world_size)
        super().__init__()

    def run(self):
        """
        process
        """
        while(True):
            recv_message(self._buff, src=0)  # 阻塞式
            sender = int(self._buff[0].item())
            message_code = MessageCode(self._buff[1].item())
            parameter = self._buff[2:]

            # need logger

            self.receive(sender, message_code, parameter)
            self.synchronise(self._worker.get_buff())

    def receive(self, sender, message_code, parameter):
        """
        开放接口
        """
        self._worker.update_model(parameter)
        self._worker.train(self.args)

    def synchronise(self, buff):
        """
        开放接口 
        """
        send_message(MessageCode.ParameterUpdate, buff)