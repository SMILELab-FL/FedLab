from multiprocessing import Lock, process
import torch.distributed as dist
from torch.functional import meshgrid
from torch.multiprocessing import Process

from fedlab_core.client.topology import ClientBasicTop
from fedlab_core.server.topology import ServerBasicTop

from fedlab_core.message_processor import MessageProcessor


class MessageQueue(object):
    def __init__(self, message_instance) -> None:
        self.write_lock = None
        self.read_lock = None

    def empty(self):
        raise NotImplementedError()

    def push(self, message):
        raise NotImplementedError()

    def front(self):
        raise NotImplementedError()

    def pop(self):
        raise NotImplementedError()


class PipeToClient(ServerBasicTop):
    """
    向下面向clinet担任mid server的角色，整合参数
    """
    def __init__(self, message_processor, server_addr, world_size, rank, dist_backend):
        super(PipeToClient, self).__init__(server_addr, world_size, rank, dist_backend)

        self.msg_processor = message_processor

        self.msg_queue_write = None
        self.msg_queue_read = None

    def run(self):
        raise NotImplementedError()

    def activate_clients(self):
        raise NotImplementedError()

    def listen_clients(self):
        raise NotImplementedError()

    def load_message_queue(self, write_queue, read_queue):
        self.msg_queue_read = read_queue
        self.msg_queue_write = write_queue


class PipeToServer(ClientBasicTop):
    """
    向topserver担任client的角色，处理和解析消息
    """

    def __init__(self, message_processor, server_addr, world_size, rank, dist_backend):
        super(PipeToServer, self).__init__(server_addr, world_size, rank, dist_backend)

        self.msg_processor = message_processor

        self.msg_queue_write = None
        self.msg_queue_read = None

    def run(self):
        return super().run()

    def on_receive(self, sender, message_code, payload):
        raise NotImplementedError()

    def synchronize(self):
        raise NotImplementedError()

    def load_message_queue(self, write_queue, read_queue):
        self.msg_queue_read = read_queue
        self.msg_queue_write = write_queue


class PipeTop(Process):
    """
    Abstract class for server Pipe topology
    simple example

    向上屏蔽局部的rank （通过全局rank到局部rank的映射解决）

    主副进程：
        server进程，用于管理本地组的子FL系统
        Pipe主进程，做消息通信和中继

    难点：进程间通信和同步
        考虑Queue 队列消息传递同步

        采用消息双队列同步和激活进程

    功能规划：
        rank映射
        子联邦合并

    """

    def __init__(self, pipe2c, pipe2s, message_queues):
        self.pipe2c = pipe2c
        self.pipe2s = pipe2s

        self.pipe2c.load_message_queue(message_queues[0], message_queues[1])
        self.pipe2s.load_message_queue(message_queues[1], message_queues[0])

    def run(self):
        """process function"""
        self.pipe2c.run()
        self.pipe2s.run()

        self.pipe2c.join()
        self.pipe2s.join()
