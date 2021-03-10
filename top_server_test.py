import torch
from fedlab_core.server.parameter_server import SSGDParameterServer
from fedlab_core.models.lenet import LeNet
from fedlab_core.server.end_top import ServerTop
from fedlab_core.utils.serialization import ravel_model_params


if __name__ == "__main__":
    model = LeNet().cpu()
    ps = SSGDParameterServer(model, client_num=2)

    top = ServerTop(ps, args=None)
    top.run()
