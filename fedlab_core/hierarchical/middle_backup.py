from _typeshed import SupportsDivMod
import threading
import sys

from torch.distributed.distributed_c10d import send

sys.path.append('/home/zengdun/FedLab/')


import torch
import torch.distributed as dist
from torch.multiprocessing import Process, Queue
torch.multiprocessing.set_sharing_strategy("file_system")

from fedlab_core.network import DistNetwork
from fedlab_core.communicator.package import Package
from fedlab_core.communicator.processor import PackageProcessor

"""
class DistNetwork(Process):
    #Manage distributed network
    def __init__(self, address, world_size, rank, dist_backend='gloo'):
        super(DistNetwork, self).__init__()
        self.address = address
        self.rank = rank
        self.world_size = world_size
        self.dist_backend = dist_backend

    def init_network_connection(self):
        print("torch.distributed initializeing processing group with ip address {}:{}, rank {}, world size: {}, backend: {}".format(self.address[0],self.address[1],self.rank, self.world_size, self.dist_backend))
        dist.init_process_group(backend=self.dist_backend,
                                init_method='tcp://{}:{}'.format(
                                    self.address[0],
                                    self.address[1]),
                                rank=self.rank,
                                world_size=self.world_size)
    
    def run(self):
        pass

    def on_receive(self, sender, message_code, payload):
        pass
"""
class ConnectClient(Process):
    """connect with clients"""
    def __init__(self, network, write_queue, read_queue):
        super(ConnectClient, self).__init__()

        self.network = network
        self.mq_read = read_queue
        self.mq_write = write_queue

        #self.rank_map = {}  # 上层rank到下层rank的映射

    def run(self):
        self.network.init_network_connection()
        # start a thread watching message queue
        watching_queue = threading.Thread(target=self.deal_queue)
        watching_queue.start()
        
        while True:
            sender, message_code, payload = PackageProcessor.recv_package()  # package from clients
            print("ConnectClient: recv data from {}, message code {}".format(sender, message_code))
            self.on_receive(sender, message_code, payload)

    def on_receive(self, sender, message_code, payload):
        self.mq_write.put((sender, message_code, payload))

    def deal_queue(self):
        """"""
        while True:
            sender, message_code, payload = self.mq_read.get()
            print("Watching Queue: data from {}, message code {}".format(sender, message_code))
            pack = Package(message_code=message_code, content=payload)
            PackageProcessor.send_package(pack, dst=1)
    
class ConnectServer(DistNetwork):
    """向topserver担任client的角色，处理和解析消息"""
    def __init__(self, network, write_queue, read_queue):
        super(ConnectServer, self).__init__()

        self.network = network
        self.mq_write = write_queue
        self.mq_read = read_queue

    def run(self):
        self.network.init_network_connection()
        # start a thread watching message queue
        watching_queue = threading.Thread(target=self.deal_queue)
        watching_queue.start()

        while True:
            sender, message_code, payload = PackageProcessor.recv_package()
            self.on_receive(sender. message_code, payload)

    def on_receive(self, sender, message_code, payload):
        print("MiddleCoodinator: recv data from {}, message code {}".format(sender, message_code))
        self.mq_write.put((sender, message_code, payload))

    def deal_queue(self):
        """ """
        while True:
            sender, message_code, payload = self.mq_read.get()
            print("data from {}, message code {}".format(sender, message_code))

            pack = Package(message_code=message_code, content=payload)
            PackageProcessor.send_package(pack, dst=0)

class MiddleServer(Process):
    """Middle Topology for hierarchical communication pattern"""
    def __init__(self):
        super(MiddleServer, self).__init__()
        self.MQs = [Queue(), Queue()]

    def run(self):
        
        cnet = DistNetwork(('127.0.0.1','3002'), world_size=2, rank=0, dist_backend="gloo")
        connect_client = ConnectClient(cnet, write_queue=self.MQs[0], read_queue=self.MQs[1])

        snet = DistNetwork(('127.0.0.1','3001'), world_size=2, rank=1, dist_backend="gloo")
        connect_server = ConnectServer(snet, write_queue=self.MQs[1], read_queue=self.MQs[0])

        connect_client.start()
        connect_server.start()

        connect_client.join()
        connect_server.join()

if __name__ == "__main__":
    middle_server = MiddleServer()
    middle_server.start()
    middle_server.join()