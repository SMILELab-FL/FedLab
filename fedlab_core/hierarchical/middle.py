import threading
import sys

from torch.distributed.distributed_c10d import send

sys.path.append('/home/zengdun/FedLab/')


import torch
import torch.distributed as dist
from torch.multiprocessing import Process, Queue
torch.multiprocessing.set_sharing_strategy("file_system")

from fedlab_core.communicator.package import Package
from fedlab_core.communicator.processor import PackageProcessor


"""
pipe top双进程采用耦合消息队列同步模式
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

class Network(Process):
    def __init__(self, server_address, world_size, rank, dist_backend='gloo'):
        super(Network, self).__init__()
        self.world_size = world_size
        self.rank = rank
        self.server_address = server_address
        self.dist_backend = dist_backend

    def init_network_connection(self):
        print("init network with ip address {}:{}".format(self.server_address[0],self.server_address[1]))
        dist.init_process_group(backend=self.dist_backend,
                                init_method='tcp://{}:{}'.format(
                                    self.server_address[0],
                                    self.server_address[1]),
                                rank=self.rank,
                                world_size=self.world_size)
    
    def run(self):
        pass

class ConnectClient(Network):
    """connect with clients"""
    def __init__(self, server_addr, world_size, rank, dist_backend, write_queue, read_queue):
        super(ConnectClient, self).__init__(
            server_addr, world_size, rank, dist_backend)
        self.mq_read = read_queue
        self.mq_write = write_queue

        #self.rank_map = {}  # 上层rank到下层rank的映射

    def run(self):
        self.init_network_connection()
        # start a thread watching message queue
        watching_queue = threading.Thread(target=self.deal_queue)
        watching_queue.start()
        
        while True:
            sender, message_code, payload = PackageProcessor.recv_package()  # package from clients
            print("ConnectClient: recv data from {}, message code {}".format(sender, message_code))
            self.receive(sender, message_code, payload)

    def deal_queue(self):
        """
        处理上层下传信息
        """
        while True:
            sender, message_code, payload = self.mq_read.get()
            print("Watching Queue: data from {}, message code {}".format(sender, message_code))
            # implement your functions
            pack = Package(message_code=message_code, content=payload)
            PackageProcessor.send_package(pack, dst=1)
    
    def receive(self, sender, message_code, payload):
        self.mq_write.put((sender, message_code, payload))

class ConnectServer(Network):
    """向topserver担任client的角色，处理和解析消息"""
    def __init__(self, server_addr, world_size, rank, dist_backend, write_queue, read_queue):
        super(ConnectServer, self).__init__(
            server_addr, world_size, rank, dist_backend)

        self.mq_write = write_queue
        self.mq_read = read_queue

    def run(self):
        self.init_network_connection()
        # start a thread watching message queue
        watching_queue = threading.Thread(target=self.deal_queue)
        watching_queue.start()

        while True:
            sender, message_code, payload = PackageProcessor.recv_package()
            self.receive(sender. message_code, payload)

    def deal_queue(self):
        """ """
        while True:
            sender, message_code, payload = self.mq_read.get()
            print("data from {}, message code {}".format(sender, message_code))

            pack = Package(message_code=message_code, content=payload)
            PackageProcessor.send_package(pack, dst=0)
        
    def receive(self, sender, message_code, payload):
        print("MiddleCoodinator: recv data from {}, message code {}".format(sender, message_code))
        self.mq_write.put((sender, message_code, payload))


class MiddleServer(Process):
    """Middle Topology for hierarchical communication pattern"""
    def __init__(self):
        super(MiddleServer, self).__init__()
        
        self.MQs = [Queue(), Queue()]

    def run(self):

        connect_client = ConnectClient(('127.0.0.1','3002'), world_size=2, rank=0, dist_backend="gloo", write_queue=self.MQs[0], read_queue=self.MQs[1])
        connect_server = ConnectServer(('127.0.0.1','3001'), world_size=2, rank=1, dist_backend="gloo", write_queue=self.MQs[1], read_queue=self.MQs[0])

        connect_client.start()
        connect_server.start()

        connect_client.join()
        connect_server.join()


if __name__ == "__main__":
    middle_server = MiddleServer()
    middle_server.start()
    middle_server.join()