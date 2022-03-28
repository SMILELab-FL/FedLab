import argparse
import sys
from torch import nn

sys.path.append("../../")
from fedlab.utils.logger import Logger
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.network import DistNetwork


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


parser = argparse.ArgumentParser(description='FL server example')

parser.add_argument('--ip', type=str)
parser.add_argument('--port', type=str)
parser.add_argument('--world_size', type=int)
parser.add_argument('--ethernet', type=str, default=None)

parser.add_argument('--round', type=int, default=3)
parser.add_argument('--sample', type=float, default=1)

args = parser.parse_args()

model = MLP()
LOGGER = Logger(log_name="server")
handler = SyncParameterServerHandler(model,
                                     global_round=args.round,
                                     logger=LOGGER,
                                     sample_ratio=args.sample)
network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=0,
                      ethernet=args.ethernet)
manager_ = ServerSynchronousManager(handler=handler,
                                    network=network,
                                    logger=LOGGER)
manager_.run()
