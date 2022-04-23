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

import torch
from torch.multiprocessing import Queue

from .connector import ClientConnector, ServerConnector
from ....utils import Logger

torch.multiprocessing.set_sharing_strategy("file_system")


class Scheduler():
    """Middle Topology for hierarchical communication pattern.
    
    Scheduler uses message queues to decouple connector modules.

    Args:
        net_upper (DistNetwork): Distributed network manager of server from upper level.
        net_lower (DistNetwork): Distributed network manager of clients from lower level.
    """

    def __init__(self, net_upper, net_lower):
        super(Scheduler, self).__init__()
        self.__MQs = [Queue(), Queue()]
        self.net_upper = net_upper
        self.logger_upper = Logger(
            log_name="Scheduler{}-ServerConnector".format(self.net_upper.rank))

        self.net_lower = net_lower
        self.logger_lower = Logger(
            log_name="Scheduler{}-ClientConnector".format(self.net_upper.rank))

    def run(self):
        connect_server = ServerConnector(self.net_upper,
                                         write_queue=self.__MQs[1],
                                         read_queue=self.__MQs[0],
                                         logger=self.logger_upper)

        connect_client = ClientConnector(self.net_lower,
                                         write_queue=self.__MQs[0],
                                         read_queue=self.__MQs[1],
                                         logger=self.logger_lower)

        connect_server.start()
        connect_client.start()

        # This is a tiny bug.
        # The process with connect_client.join() is always blocked somehow in the shutdown stage.
        # You are will come to solve this. (I think it's some process synchronization problems)
        connect_server.join()
        # connect_client.join()
        connect_client.kill()
