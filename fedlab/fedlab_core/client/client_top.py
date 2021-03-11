
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab.fedlab_core.utils.messaging import send_message, recv_message, MessageCode


class ClientTop(Process):
    def __init__(self, worker, args):
        self._worker = worker
        self._buff = torch.zeros(worker.get_buff().numel() + 2).cpu()

        dist.init_process_group(backend="gloo", init_method='tcp://{}:{}'
                                .format(args.server_ip, args.server_port),
                                rank=args.rank, world_size=args.world_size)
        #self.client_store = dist.TCPStore(args.server_ip, args.server_port, args.world_size, False)
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

            self.receive(sender, message_code, parameter)
            self.sync(self._worker.get_buff())

    def receive(self, sender, message_code, parameter):
        """
        开放接口
        """
        self._worker.update_model(parameter)
        self._worker.train(self._buff)

    def sync(self, buff):
        """
        开放接口 
        """
        send_message(MessageCode.ParameterUpdate, buff)


if __name__ == "__main__":
    print("test")
