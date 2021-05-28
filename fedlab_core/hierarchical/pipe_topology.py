
from queue import Queue
import torch
import torch.distributed as dist
from torch.functional import meshgrid
from torch.multiprocessing import Process

from ...fedlab_core.client.topology import ClientBasicTop
from ...fedlab_core.server.topology import ServerBasicTop
from ...fedlab_utils.serialization import SerializationTool
from ...fedlab_utils.message_code import MessageCode
from ...fedlab_core.communicator import Package


"""
pipe top双进程采用低耦合消息队列同步模式
    启动流程:
        1. Pipe2Client(下称进程P2C)开启接受参与者网络请求，构成下层group  
           Pipe2Server(下称进程P2S)与上层server通信，构成上层group
        2. P2C与client连接成功后，向管道发送网络包 MessageCode.Registration，汇报当前组Worker数量
           P2S收到P2C的Registration，向上层server转发汇报，与上层Server通信构建转发表（worker_id -> group_rank）
        3. 开启联邦学习进程：
            server向下层通信[client_list, parameters]
            pipeTop 做id->rank映射包转发
            client将模型上传pipe
            pipeTop 做合并 mid model并上传[parameters, client_num]
            server获得mid model合并 global model  
"""

class MessageQueue(object):
    def __init__(self) -> None:
        self.MQ = Queue(maxsize=10)

    def empty(self):
        return self.MQ.empty()

    def put(self, package):
        self.MQ.put(package)

    def get(self):
        return self.MQ.get(block=True)


class PipeToClient(ServerBasicTop):
    """
    向下面向clinet担任mid server的角色，整合参数
    """

    def __init__(self, server_addr, world_size, rank, dist_backend):
        super(PipeToClient, self).__init__(
            server_addr, world_size, rank, dist_backend)

        self.mq_read = None
        self.mq_write = None

        self.model_cache = torch.zeros(
            size=(self.msg_processor.serialized_param_size)).cpu()
        self.cache_cnt = 0

        self.rank_map = {}  # 上层rank到下层rank的映射

    def run(self):
        if self.mq_write is None or self.mq_read is NotImplemented:
            raise ValueError("invalid MQs")
        self.init_network_connection()

        while True:
            message = self.mq_read.get()
            sender, recver, header, parameters = self.msg_processor.unpack(
                message)
            message_code = header
            if message_code == MessageCode.ParameterUpdate:
                self.model_cache += parameters
                self.cache_cnt += 1

    def activate_clients(self, payload):
        clients_this_round = []
        for client_idx in clients_this_round:
            payload = payload
            self.msg_processor.send_package(payload=payload, dst=client_idx)

    def listen_clients(self):
        package = self.msg_processor.recv_package()
        sender, recver, header, parameters = self.msg_processor.unpack(
            payload=package)

    def load_message_queue(self, write_queue, read_queue):
        self.mq_read = read_queue
        self.mq_write = write_queue


class PipeToServer(ClientBasicTop):
    """
    向topserver担任client的角色，处理和解析消息
    """
    def __init__(self, server_addr, world_size, rank, dist_backend):
        super(PipeToServer, self).__init__(
            server_addr, world_size, rank, dist_backend)

        self.mq_write = None
        self.mq_read = None

    def run(self):
        if self.mq_write is None or self.mq_read is NotImplemented:
            raise ValueError("invalid MQs")
        self.init_network_connection()

    def on_receive(self, sender, message_code, payload):
        raise NotImplementedError()

    def synchronize(self):
        raise NotImplementedError()

    def load_message_queue(self, write_queue, read_queue):
        self.mq_read = read_queue
        self.mq_write = write_queue


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

    def __init__(self, pipe2c, pipe2s):
        self.pipe2c = pipe2c
        self.pipe2s = pipe2s

        self.MQs = [MessageQueue(), MessageQueue()]
        self.pipe2c.load_message_queue(self.MQs[0], self.MQs[1])
        self.pipe2s.load_message_queue(self.MQs[1], self.MQs[0])

    def run(self):
        """process function"""
        self.pipe2c.run()
        self.pipe2s.run()

        self.pipe2c.join()
        self.pipe2s.join()
