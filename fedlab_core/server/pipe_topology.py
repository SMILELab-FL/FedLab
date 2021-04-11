from multiprocessing import Lock
import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab_core.utils.messaging import MessageCode, recv_message, send_message

from fedlab_core.client.topology import ClientCommunicationTopology
from fedlab_core.server.handler import SyncParameterServerHandler
from fedlab_core.server.topology import EndTop, ServerBasicTop


class PipeTop(ClientCommunicationTopology):
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
    def __init__(self, model, args):

        self._model = model
        self._model.share_memory_()

    def run(self):
        """process function"""
        raise NotImplementedError()

    def on_receive(self, sender, message_code, payload):
        """
        接收上层server的激活信息
        """
        return super().receive(sender, message_code, payload)

    def synchronise(self, payload):
        """
        将本组内的局部合并后的模型上传上层服务器
        """
        return super().synchronise(payload)