import sys

sys.path.append('../../../')

import torch
from torch.multiprocessing import Process, Queue
torch.multiprocessing.set_sharing_strategy("file_system")

from fedlab_core.hierarchical.connector import ConnectClient, ConnectServer


class Scheduler(Process):
    """Middle Topology for hierarchical communication pattern
    
        scheduler use message queues to decouple connector modules
    """
    def __init__(self, net_upper, net_lower):
        super(Scheduler, self).__init__()
        self.MQs = [Queue(), Queue()]
        self.net_upper = net_upper
        self.net_lower = net_lower

    def run(self):

        connect_client = ConnectClient(self.net_lower,
                                       write_queue=self.MQs[0],
                                       read_queue=self.MQs[1])
        connect_server = ConnectServer(self.net_upper,
                                       write_queue=self.MQs[1],
                                       read_queue=self.MQs[0])

        connect_client.start()
        connect_server.start()

        connect_client.join()
        connect_server.join()