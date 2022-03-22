
import argparse
from fedlab.core.network import DistNetwork

parser = argparse.ArgumentParser(description="Network connection checker")

parser.add_argument("--ip", type=str)
parser.add_argument("--port", type=str)
parser.add_argument("--world_size", type=int)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--ethernet", type=str, default=None)
args = parser.parse_args()

network = DistNetwork(
    address=(args.ip, args.port),
    world_size=args.world_size,
    rank=args.rank,
    ethernet=args.ethernet,
)

print("Network connection initializing...")
network.init_network_connection()
print("Network connection created!")
network.close_network_connection()
print("Network connection test finished!")
print("Current configuration is OK!")
