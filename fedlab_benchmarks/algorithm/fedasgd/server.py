import sys
sys.path.append('../../../')
# sys.path.append('/home/zengdun/FedLab/')

from models.lenet import LeNet
from fedlab_core.server.handler import AsyncParameterServerHandler
from fedlab_core.server.topology import ServerAsyncTop
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default='3002')
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()

    model = LeNet().cpu()
    ps = AsyncParameterServerHandler(
        model, client_num_in_total=args.world_size-1)  # client = world_size-1
    top = ServerAsyncTop(server_handler=ps, server_address=(
        args.server_ip, args.server_port))
    top.run()
