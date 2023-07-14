import argparse
import sys

sys.path.append("../../")

from fedlab.core.network import DistNetwork
from fedlab.models.mlp import MLP
from fedlab.board import fedboard
from fedlab.board.utils.roles import SERVER
from pipeline.server_side import ExampleHandler, ExampleManager

parser = argparse.ArgumentParser(description='FL server example')

parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=str, default="3002")
parser.add_argument('--world_size', type=int)
parser.add_argument('--ethernet', type=str, default=None)

parser.add_argument('--round', type=int)
parser.add_argument('--sample', type=float, default=0.5)

args = parser.parse_args()

model = MLP(784, 10)

handler = ExampleHandler(model,
                         global_round=args.round,
                         sample_ratio=args.sample,

                         cuda=False)

network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=0)

manager_ = ExampleManager(network=network, handler=handler, mode="GLOBAL")

fedboard.register(id='mtp-01', max_round=args.round, roles=SERVER)

manager_.run()
