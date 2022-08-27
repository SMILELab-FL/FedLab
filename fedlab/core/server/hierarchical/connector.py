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
from time import sleep
import torch

from torch.multiprocessing import Queue
from ...network import DistNetwork
from ...network_manager import NetworkManager
from ...communicator.processor import PackageProcessor
from ...coordinator import Coordinator
from ....utils import MessageCode,Logger

torch.multiprocessing.set_sharing_strategy("file_system")


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

    def __init__(self, network: DistNetwork, write_queue: Queue,
                 read_queue: Queue):
        super(Connector, self).__init__(network)

        # bidirectional message queues
        self.mq_read = read_queue
        self.mq_write = write_queue

    def process_meessage_queue(self):
        """Define the procedure of dealing with message queue."""
        raise NotImplementedError()


class ServerConnector(Connector):
    """Connect with server.

        this process will act like a client.

        This class is a part of middle server which used in hierarchical structure.

    Args:
        network (DistNetwork): Network configuration and interfaces.
        write_queue (torch.multiprocessing.Queue): Message queue to write.
        read_queue (torch.multiprocessing.Queue):  Message queue to read.
        logger (Logger, optional): object of :class:`Logger`. Defaults to None.
    """

    def __init__(self,
                 network: DistNetwork,
                 write_queue: Queue,
                 read_queue: Queue,
                 logger: Logger = None):
        super(ServerConnector, self).__init__(network, write_queue, read_queue)

        self.mq_write = write_queue
        self.mq_read = read_queue

        self.group_client_num = 0
        self._LOGGER = Logger() if logger is None else logger

    def run(self):
        """
        Main Process:

          1. Initialization stage.
          2. FL communication stage.
          3. Shutdown stage. Close network connection.
        """
        self.setup()
        self.main_loop()
        self.shutdown()

    def setup(self):
        super().setup()

        _, message_code, payload = self.mq_read.get()
        assert message_code == MessageCode.SetUp

        self.group_client_num = payload[0].item()
        self._network.send(content=torch.Tensor([self.group_client_num]).int(),
                           message_code=message_code,
                           dst=0)

    def main_loop(self):
        # start a thread watching message queue
        watching_queue = threading.Thread(target=self.process_meessage_queue,
                                          daemon=True)
        watching_queue.start()

        while True:
            # server -> client
            sender, message_code, payload = PackageProcessor.recv_package()
            self.mq_write.put_nowait((sender, message_code, payload))

            if message_code == MessageCode.Exit:
                # client exit feedback
                if self._network.rank == self._network.world_size - 1:
                    self._network.send(message_code=MessageCode.Exit, dst=0)

                self._LOGGER.info("the main loop exit.")

                watching_queue.join()
                break

    def process_meessage_queue(self):
        """ client -> server
            directly transport.
        """
        while True:
            sender, message_code, payload = self.mq_read.get()
            self._LOGGER.info(
                "[Queue-Thread: client -> server] recv data from rank {}, message code {}."
                .format(sender, message_code))

            if message_code == MessageCode.Exit:
                self._LOGGER.info(
                    "[Queue-Thread] process_meessage_queue thread exit.")
                break

            self._network.send(content=payload,
                               message_code=message_code,
                               dst=0)


class ClientConnector(Connector):
    """Connect with clients.

    This class is a part of middle server which used in hierarchical structure.

    Args:
        network (DistNetwork): Network configuration and interfaces.
        write_queue (torch.multiprocessing.Queue): Message queue to write.
        read_queue (torch.multiprocessing.Queue):  Message queue to read.
        logger (Logger, optional): object of :class:`Logger`. Defaults to None.
    """

    def __init__(self,
                 network: DistNetwork,
                 write_queue: Queue,
                 read_queue: Queue,
                 logger: Logger = None):
        super(ClientConnector, self).__init__(network, write_queue, read_queue)

        self.mq_read = read_queue
        self.mq_write = write_queue

        self.coordinator = None
        self.group_client_num = 0

        self._LOGGER = Logger() if logger is None else logger

    def run(self):
        """
        Main Process:

          1. Initialization stage.
          2. FL communication stage.
          3. Shutdown stage. Close network connection.
        """
        self.setup()
        self.main_loop()
        self.shutdown(
        )  # There is a bug. ClientConnector process will be blocked here. TODO:unsolved.

    def setup(self):
        super().setup()
        rank_client_id_map = {}

        for rank in range(1, self._network.world_size):
            _, _, content = self._network.recv(src=rank)
            rank_client_id_map[rank] = content[0].item()
        self.coordinator = Coordinator(rank_client_id_map)

        self.group_client_num = self.coordinator.total

        self.mq_write.put_nowait((self._network.rank, MessageCode.SetUp,
                                  torch.Tensor([self.group_client_num]).int()))

    def main_loop(self):
        # start a thread to watch message queue
        watching_queue = threading.Thread(target=self.process_meessage_queue,
                                          daemon=True)
        watching_queue.start()

        while True:
            sender, message_code, payload = self._network.recv(
            )  # unexpected block. TODO: fix this.

            if message_code == MessageCode.Exit:
                self._LOGGER.info("main loop exit.")
                self.mq_write.put_nowait((None, MessageCode.Exit, None))

                watching_queue.join()
                break

            self._LOGGER.info(
                "[client -> server] recv data from rank {}, message code {}.".
                format(sender, message_code))
            self.mq_write.put_nowait((sender, message_code, payload))

    def process_meessage_queue(self):
        """Process message queue

        Strategy of processing message from server.

        """
        while True:
            # server -> client
            sender, message_code, payload = self.mq_read.get()
            self._LOGGER.info(
                "[Queue-Thread: server -> client] recv data from rank {}, message code {}."
                .format(sender, message_code))

            # broadcast message
            id_list, payload = payload[0].to(torch.int32).tolist(), payload[1:]
            rank_dict = self.coordinator.map_id_list(id_list)

            for rank, values in rank_dict.items():
                id_list = torch.Tensor(values).to(torch.int32)
                self._network.send(content=[id_list] + payload,
                                   message_code=message_code,
                                   dst=rank)

            if message_code == MessageCode.Exit:
                sleep(5)
                self._LOGGER.info(
                    "[Queue-Thread] process_meessage_queue thread exit.")
                break
