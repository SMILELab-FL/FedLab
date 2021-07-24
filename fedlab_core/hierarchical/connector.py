import threading
import sys

sys.path.append('../../../')

import torch
from torch.multiprocessing import Queue

torch.multiprocessing.set_sharing_strategy("file_system")

from fedlab_core.network import DistNetwork
from fedlab_core.communicator.package import Package
from fedlab_core.communicator.processor import PackageProcessor
from fedlab_core.network_manager import NetworkManager


class Connector(NetworkManager):
    """Abstract class for basic Connector, which is a sub-module of schedular.

    Args:
        newtork (`DistNetwork`): object to manage torch.distributed network communication.
        write_queue (Queue): message queue
        read_queue (Queue):  message queue
    """
    def __init__(self, network: DistNetwork, write_queue: Queue,
                 read_queue: Queue):
        super(Connector, self).__init__(network)

        self.mq_read = read_queue
        self.mq_write = write_queue

    def run(self):
        return super().run()

    def on_receive(self, sender, message_code, payload):
        """define the reaction of receiving message.

        Args:
            sender (int): rank of sender in dist group.
            message_code (`MessageCode`): message code
            payload (`torch.Tensor`): Tensors 
        """
        pass

    def deal_queue():
        pass


class ConnectClient(Connector):
    """Connect with clients.

        This class is a part of middle server which used in hierarchical structure.

        TODO: middle server
        
    Args:
        newtork (`DistNetwork`): object to manage torch.distributed network communication.
        write_queue (Queue): message queue
        read_queue (Queue):  message queue
    """
    def __init__(self, network: DistNetwork, write_queue: Queue,
                 read_queue: Queue):
        super(ConnectClient, self).__init__(network, write_queue, read_queue)

        self.mq_read = read_queue
        self.mq_write = write_queue

    def run(self):
        self._network.init_network_connection()
        # start a thread to watch message queue
        watching_queue = threading.Thread(target=self.deal_queue)
        watching_queue.start()

        while True:
            sender, message_code, payload = PackageProcessor.recv_package(
            )  # package from clients
            print("ConnectClient: recv data from {}, message code {}".format(
                sender, message_code))
            self.on_receive(sender, message_code, payload)

    def on_receive(self, sender, message_code, payload):
        self.mq_write.put((sender, message_code, payload))

    def deal_queue(self):
        """Process message queue"""
        while True:
            sender, message_code, payload = self.mq_read.get()
            print("Watching Queue: data from {}, message code {}".format(
                sender, message_code))
            pack = Package(message_code=message_code, content=payload)
            PackageProcessor.send_package(pack, dst=1)


class ConnectServer(Connector):
    """Connect with server.

        This class is a part of middle server which used in hierarchical structure.
        
        TODO:Rank mapper

    Args:
        newtork (`DistNetwork`): object to manage torch.distributed network communication.
        write_queue (Queue): message queue
        read_queue (Queue):  message queue
    """
    def __init__(self, network: DistNetwork, write_queue: Queue,
                 read_queue: Queue):
        super(ConnectServer, self).__init__(network, write_queue, read_queue)

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