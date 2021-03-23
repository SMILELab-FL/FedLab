import sys
sys.path.append('/home/zengdun/FedLab')

from fedlab_core.server.handler import SyncSGDParameterServerHandler
from fedlab_core.models.lenet import LeNet
from fedlab_core.server.topology import EndTop

if __name__ == "__main__":
    model = LeNet().cpu()
    ps = SyncSGDParameterServerHandler(model, client_num=2)
    top = EndTop(ps, server_addr=('127.0.0.1', '3001'), args=None)
    top.run()
