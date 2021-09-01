# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch.distributed as dist


class DistNetwork(object):
    """Manage ``torch.distributed`` network

    Args:
        address (tuple): Address of this server in form of ``(SERVER_ADDR, SERVER_IP)``
        world_size (int): the size of this distributed group (including server).
        rank (int): the rank of process in distributed group.
        ethernet (str)
        dist_backend (str or torch.distributed.Backend): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``, and ``nccl``. Default: ``"gloo"``.
    """

    def __init__(self,
                 address,
                 world_size,
                 rank,
                 ethernet=None,
                 dist_backend="gloo"):
        super(DistNetwork, self).__init__()
        self.address = address
        self.rank = rank
        self.world_size = world_size
        self.dist_backend = dist_backend
        self.ethernet = ethernet

    def init_network_connection(self):
        """Initialize ``torch.distributed`` communication group"""
        print(self.__str__())

        if self.ethernet is not None:
            os.environ["GLOO_SOCKET_IFNAME"] = self.ethernet

        dist.init_process_group(
            backend=self.dist_backend,
            init_method="tcp://{}:{}".format(self.address[0], self.address[1]),
            rank=self.rank,
            world_size=self.world_size,
        )

    def close_network_connection(self):
        """Destroy current ``torch.distributed`` process group"""
        if dist.is_initialized():
            dist.destroy_process_group()

    def __str__(self):
        info_str = "torch.distributed connection is initializing with server ip address {}:{}, rank {}, world size: {}, backend {}, ethernet {}.".format(
            self.address[0],
            self.address[1],
            self.rank,
            self.world_size,
            self.dist_backend,
            self.ethernet,
        )
        return info_str
