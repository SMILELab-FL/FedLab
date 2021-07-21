import threading
import sys

from torch.distributed.distributed_c10d import send

sys.path.append('/home/zengdun/FedLab/')
import torch
from torch.multiprocessing import Queue, Process
from fedlab_core.network import DistNetwork
from fedlab_core.hierarchical.scheduler import Scheduler
torch.multiprocessing.set_sharing_strategy("file_system")


"""
class Scheduler(Process):
    #Middle Topology for hierarchical communication pattern
    def __init__(self):
        super(Scheduler, self).__init__()
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
"""

if __name__ == "__main__":
    cnet = DistNetwork(('127.0.0.1','3002'), world_size=2, rank=0, dist_backend="gloo")
    snet= DistNetwork(('127.0.0.1','3001'), world_size=2, rank=1, dist_backend="gloo")
    middle_server = Scheduler(net_upper=snet, net_lower=cnet)
    middle_server.start()
    middle_server.join()