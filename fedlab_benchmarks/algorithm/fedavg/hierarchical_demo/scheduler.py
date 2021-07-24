import sys

from torch.distributed.distributed_c10d import send

sys.path.append('../../../')
from fedlab_core.network import DistNetwork
from fedlab_core.hierarchical.scheduler import Scheduler


if __name__ == "__main__":
    cnet = DistNetwork(('127.0.0.1','3002'), world_size=2, rank=0, dist_backend="gloo")
    snet= DistNetwork(('127.0.0.1','3001'), world_size=2, rank=1, dist_backend="gloo")
    middle_server = Scheduler(net_upper=snet, net_lower=cnet)
    middle_server.start()
    middle_server.join()