import sys
sys.path.append("/home/zengdun/FedLab")

import torch.distributed as dist
from torch.multiprocessing import Process

from models.lenet import LeNet
from fedlab_core.communicator.processor import Package, PackageProcessor

class recver(Process):
    def run(self):
        print("server waiting")
        dist.init_process_group(backend='gloo', init_method='tcp://{}:{}'
                                .format('127.0.0.1', '3000'),
                                rank=0, world_size=2)
        print("server connected")
        mc, sr, content = PackageProcessor.recv_package()
        print(content)
        for data in content:
            print(data.shape)

if __name__ == "__main__":
    p = recver()
    p.run()