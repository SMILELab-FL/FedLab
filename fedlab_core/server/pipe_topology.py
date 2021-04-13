from multiprocessing import Lock
import torch.distributed as dist
from torch.functional import meshgrid
from torch.multiprocessing import Process

from fedlab_core.client.topology import ClientBasicTop
from fedlab_core.server.topology import ServerBasicTop

from fedlab_core.message_processor import MessageProcessor


class PipeToServer(ServerBasicTop):
    """
    """

    def __init__(self, msg_processor, locks, server_handler, server_address, dist_backend, logger_path, logger_name):
        super().__init__(server_handler, server_address, dist_backend=dist_backend,
                         logger_path=logger_path, logger_name=logger_name)

        self.msg_processor = msg_processor
        self.locks = locks


class PipeToClient(ClientBasicTop):
    """

    """

    def __init__(self, msg_processor, locks, backend_handler, server_addr, world_size, rank, dist_backend):
        super().__init__(backend_handler, server_addr, world_size, rank, dist_backend)

        self.msg_processor = msg_processor
        self.locks = locks


class PipeTop(ClientBasicTop):
    """
    Abstract class for server Pipe topology
    simple example


    向上屏蔽局部的rank （通过全局rank到局部rank的映射解决）


    双进程：
        server进程，用于管理本地组的子FL系统
        Pipe主进程，做消息通信和中继

    难点：进程间通信和同步
        考虑Queue 队列消息传递同步

    """

    def __init__(self, backend_handler, server_addr, world_size, rank, dist_backend):
        super().__init__(backend_handler, server_addr, world_size, rank, dist_backend)

        self.msg_processor = MessageProcessor(
            control_code_size=2, model=backend_handler.model)

    def run(self):
        """process function"""
        pass

    def on_receive(self, sender, message_code, payload):
        """
        接收上层server的模型信息和id
        """
        return super().receive(sender, message_code, payload)

    def synchronise(self, payload):
        """
        将本组内的局部合并后的模型上传上层服务器
        """
        return super().synchronise(payload)
