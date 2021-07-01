import threading

from queue import Queue
import torch
import torch.distributed as dist
from torch.functional import meshgrid
from torch.multiprocessing import Process

from fedlab_core.communicator.package import Package
from fedlab_utils.serialization import SerializationTool
from fedlab_utils.message_code import MessageCode
from fedlab_core.communicator.processor import PackageProcessor


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

class Network(object):
    def __init__(self, server_address, world_size, rank, dist_backend='gloo'):
        self.world_size = world_size
        self.rank = rank
        self.server_address = server_address
        self.dist_backend = dist_backend

    def init_network_connection(self):
        dist.init_process_group(backend=self.dist_backend,
                                init_method='tcp://{}:{}'.format(
                                    self.server_address[0],
                                    self.server_address[1]),
                                rank=self.rank,
                                world_size=self.world_size)


class PipeToClient(Network):
    """
    向下面向clinet担任mid server的角色，整合参数
    """
    def __init__(self, server_addr, world_size, rank, dist_backend, write_queue, read_queue):
        super(PipeToClient, self).__init__(
            server_addr, world_size, rank, dist_backend)

        self.mq_read = read_queue
        self.mq_write = write_queue

        #self.rank_map = {}  # 上层rank到下层rank的映射

    def run(self):
        self.init_network_connection()
        while True:
            sender, message_code, payload = PackageProcessor.recv_package()
            """
            deal with information hear
            """
            self.mq_write.put((sender, message_code, payload))

    def deal_queue(self):
        """
        处理上层下传信息
        """
        sender, message_code, payload = self.mq_read.get()
        print("data from {}, message code {}".format(sender, message_code))
        # implement your functions
        """
        """
        

class PipeToServer(Network):
    """向topserver担任client的角色，处理和解析消息
   
    """
    def __init__(self, server_addr, world_size, rank, dist_backend, write_queue, read_queue):
        super(PipeToServer, self).__init__(
            server_addr, world_size, rank, dist_backend)

        self.mq_write = write_queue
        self.mq_read = read_queue

    def run(self):
        self.init_network_connection()
        while True:
            sender, message_code, payload = PackageProcessor.recv_package()
            """
            deal with information from server
            """
            self.mq_write.put((sender, message_code, payload))

    def deal_queue(self):
        """
        处理下层向上传的信息
        """
        sender, message_code, payload = self.mq_read.get()
        print("data from {}, message code {}".format(sender, message_code))

        pack = Package(message_code=message_code, content=payload)
        PackageProcessor.send_package(pack, dst=0)
        
class MiddleTopology(Process):
    """Middle Topology for hierarchical communication pattern

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

    def run(self):
        """process function"""
        self.pipe2c.run()
        self.pipe2s.run()

        self.pipe2c.join()
        self.pipe2s.join()
