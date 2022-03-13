# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading

import torch
torch.multiprocessing.set_sharing_strategy("file_system")

from ...network_manager import NetworkManager
from ...communicator.processor import PackageProcessor
from ...communicator.package import Package



class Connector(NetworkManager):
    """Abstract class for basic Connector, which is a sub-module of :class:`Scheduler`.

    Connector inherits :class:`NetworkManager`, maintaining two Message Queue.
    One is for sending messages to collaborator, the other is for read messages from others.

    Note:
        Connector is a basic component for scheduler, Example code can be seen in ``scheduler.py``.

    Args:
        network (DistNetwork): Manage ``torch.distributed`` network communication.
        write_queue (torch.multiprocessing.Queue): Message queue to write.
        read_queue (torch.multiprocessing.Queue):  Message queue to read.
    """
    def __init__(self, network, write_queue, read_queue):
        super(Connector, self).__init__(network)

        self.mq_read = read_queue
        self.mq_write = write_queue

    def run(self):
        raise NotImplementedError()

    def on_receive(self, sender, message_code, payload):
        """Define the reaction of receiving message.

        Args:
            sender (int): rank of sender in dist group.
            message_code (MessageCode): Message code.
            payload (list(torch.Tensor)): A list of tensors received from other process.
        """
        raise NotImplementedError()

    def deal_queue(self):
        """Define the procedure of dealing with message queue."""
        raise NotImplementedError()


class ClientConnector(Connector):
    """Connect with clients.

    This class is a part of middle server which used in hierarchical structure.

    Args:
        network (DistNetwork): Manage ``torch.distributed`` network communication.
        write_queue (torch.multiprocessing.Queue): Message queue to write.
        read_queue (torch.multiprocessing.Queue):  Message queue to read.
    """
    def __init__(self, network, write_queue, read_queue):
        super(ClientConnector, self).__init__(network, write_queue, read_queue)

        self.mq_read = read_queue
        self.mq_write = write_queue

    def run(self):
        self.setup()
        # start a thread to watch message queue
        watching_queue = threading.Thread(target=self.deal_queue)
        watching_queue.start()

        while True:
            sender, message_code, payload = PackageProcessor.recv_package(
            )  # package from clients
            print("ClientConnector: recv data from {}, message code {}".format(
                sender, message_code))
            self.on_receive(sender, message_code, payload)

    def on_receive(self, sender, message_code, payload):
        self.mq_write.put((sender, message_code, payload))

    def deal_queue(self):
        """Process message queue

        Strategy of processing message from server.
        """
        while True:
            sender, message_code, payload = self.mq_read.get()
            print("Watching Queue: data from {}, message code {}".format(
                sender, message_code))
            pack = Package(message_code=message_code, content=payload)
            PackageProcessor.send_package(pack, dst=1)


class ServerConnector(Connector):
    """Connect with server.

    This class is a part of middle server which used in hierarchical structure.

    Args:
        network (DistNetwork): object to manage torch.distributed network communication.
        write_queue (torch.multiprocessing.Queue): message queue
        read_queue (torch.multiprocessing.Queue):  message queue
    """
    def __init__(self, network, write_queue, read_queue):
        super(ServerConnector, self).__init__(network, write_queue, read_queue)

        self.mq_write = write_queue
        self.mq_read = read_queue

    def run(self):
        self.setup()
        # start a thread watching message queue
        watching_queue = threading.Thread(target=self.deal_queue)
        watching_queue.start()

        while True:
            sender, message_code, payload = PackageProcessor.recv_package()
            self.on_receive(sender, message_code, payload)

    def on_receive(self, sender, message_code, payload):
        self.mq_write.put((sender, message_code, payload))

    def deal_queue(self):
        while True:
            sender, message_code, payload = self.mq_read.get()
            print("data from {}, message code {}".format(sender, message_code))

            pack = Package(message_code=message_code, content=payload)
            PackageProcessor.send_package(pack, dst=0)
