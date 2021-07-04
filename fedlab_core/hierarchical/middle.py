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
from fedlab_core.topology import Topology

class ConnectClient(Topology):
    """Connect with clients.

        This class is a part of middle server which used in hierarchical structure.

        TODO: middle server
        
    Args:
        newtork (`DistNetwork`): object to manage torch.distributed network communication.
        write_queue (Queue): message queue
        read_queue (Queue):  message queue
    """
    def __init__(self, network, write_queue, read_queue):
        super(ConnectClient, self).__init__(None, network)

        #self._network = network
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
        """Process message queue"""
        while True:
            sender, message_code, payload = self.mq_read.get()
            print("Watching Queue: data from {}, message code {}".format(sender, message_code))
            pack = Package(message_code=message_code, content=payload)
            PackageProcessor.send_package(pack, dst=1)
    
class ConnectServer(Topology):
    """Connect with server.

        This class is a part of middle server which used in hierarchical structure.
        
        TODO:Rank mapper

    Args:
        newtork (`DistNetwork`): object to manage torch.distributed network communication.
        write_queue (Queue): message queue
        read_queue (Queue):  message queue
    """

    def __init__(self, network, write_queue, read_queue):
        super(ConnectServer, self).__init__(None, network)

        #self._network = network
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
        self.mq_write.put((sender, message_code, payload))

    def deal_queue(self):
        """Process message queue"""
        while True:
            sender, message_code, payload = self.mq_read.get()
            print("data from {}, message code {}".format(sender, message_code))

            pack = Package(message_code=message_code, content=payload)
            PackageProcessor.send_package(pack, dst=0)