import torch.distributed as dist
from torch.multiprocessing import Process


class DistNetwork(object):
    """Manage torch.distributed network
    
    Args:
        address (tuple):
        world_size ():
        rank ():
        dist_backend ():
    
    """
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
    
    def show_configuration(self):
        pass