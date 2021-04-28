import sys
import torch
sys.path.append("/home/zengdun/FedLab")

import torch.distributed as dist
from torch.multiprocessing import Process

from models.lenet import LeNet
from fedlab_core.communicator.processor import Package, PackageProcessor
from fedlab_utils.serialization import SerializationTool


class sender(Process):
    def run(self):
        print("client connect")
        dist.init_process_group(backend='gloo', init_method='tcp://{}:{}'
                                .format('127.0.0.1', '3000'),
                                rank=1, world_size=2)
        print("connect done")

        pack = Package()
        
        random_t = torch.rand(size=(10,))
        model_t = SerializationTool.serialize_model(LeNet())

        pack.append_tensor_list([random_t, model_t])

        PackageProcessor.send_package(pack, dst=0)
        

if __name__ == "__main__":
    p = sender()
    p.run()
   
