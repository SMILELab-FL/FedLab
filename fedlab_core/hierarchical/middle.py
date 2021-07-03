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

class ConnectClient(Process):
    """Connect with clients.

        This class is a part of middle server which used in hierarchical structure.

        TODO: middle server
        
    Args:
        network (DistNetwork): 
        write_queue (Queue): message queue
        read_queue (Queue):  message queue
    """
    def __init__(self, network, write_queue, read_queue):
        super(ConnectClient, self).__init__()

        self._network = network
        self.mq_read = read_queue
        self.mq_write = write_queue

        #self.rank_map = {}  # 上层rank到下层rank的映射

    def run(self):
        self._network.init_network_connection()
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
    
class ConnectServer(Process):
    """Connect with server.

        This class is a part of middle server which used in hierarchical structure.
        
        TODO:Rank mapper

    Args:
        network (DistNetwork): 
        write_queue (Queue): message queue
        read_queue (Queue):  message queue
    """

    def __init__(self, network, write_queue, read_queue):
        super(ConnectServer, self).__init__()

        self._network = network
        self.mq_write = write_queue
        self.mq_read = read_queue

    def run(self):
        self._network.init_network_connection()
        # start a thread watching message queue
        watching_queue = threading.Thread(target=self.deal_queue)
        watching_queue.start()

        while True:
            sender, message_code, payload = PackageProcessor.recv_package()
            self.on_receive(sender, message_code, payload)

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

        snet= DistNetwork(('127.0.0.1','3001'), world_size=2, rank=1, dist_backend="gloo")
        connect_server = ConnectServer(snet, write_queue=self.MQs[1], read_queue=self.MQs[0])

        connect_client.start()
        connect_server.start()

        connect_client.join()
        connect_server.join()

if __name__ == "__main__":
    middle_server = MiddleServer()
    middle_server.start()
    middle_server.join()