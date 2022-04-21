from cgitb import handler
import os
import sys
import argparse
from torch import nn
sys.path.append("../../")
from fedlab.core.network import DistNetwork
from fedlab.core.server.handler import AsyncParameterServerHandler
from fedlab.core.server.manager import AsynchronousServerManager


# torch model
class MLP(nn.Module):
    def __init__(self, input_size=784, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    model = MLP()
    handler = AsyncParameterServerHandler(model, alpha=0.5, total_time=5)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)
    Manager = AsynchronousServerManager(handler=handler, network=network)

    Manager.run()
