
from fedlab_core.server.pshandler import SSGDParameterServer
from fedlab_core.models.lenet import LeNet
from fedlab_core.server.topology import ServerTop

if __name__ == "__main__":
    model = LeNet().cpu()
    ps = SSGDParameterServer(model, client_num=2)
    top = ServerTop(ps, args=None)
    top.run()
