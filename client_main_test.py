import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

from FLTB_core.models.lenet import LeNet
from FLTB_core.utils.messaging import recv_message, send_message, MessageCode
from FLTB_core.utils.serialization import ravel_model_params, unravel_model_params



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--server_ip', type=str, default="127.0.0.1")
    parser.add_argument('--server_port', type=int, default=3001)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--world_size', type=int, default=3)
    args = parser.parse_args()

    dist.init_process_group('gloo', init_method='tcp://{}:{}'
                            .format(args.server_ip, args.server_port),
                            rank=args.local_rank, world_size=args.world_size)

    model = LeNet()
    buff = ravel_model_params(model)

    message_code = 1
    local_buff = torch.Tensor([dist.get_rank(), message_code])
    local_buff = torch.cat((local_buff, buff.cpu()))

    model_buff = recv_message(local_buff, src=0)
    unravel_model_params(model, model_buff)

    # train process
    buff[:] = 0

    local_buff = send_message(MessageCode.ParameterUpdate, buff, dst=0)
