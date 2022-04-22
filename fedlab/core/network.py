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
import torch
import torch.distributed as dist

from .communicator.processor import Package, PackageProcessor
from ..utils import Logger

type2byte = {
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.float16: 2,
    torch.float32: 4,
    torch.float64: 8
}


class DistNetwork(object):
    """Manage ``torch.distributed`` network.

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
        self._LOGGER = Logger(log_name="network {}".format(self.rank))

        self.send_volume_intotal = 0  # byte
        self.recv_volume_intotal = 0  # byte

    def init_network_connection(self):
        """Initialize ``torch.distributed`` communication group"""
        self._LOGGER.info(self.__str__())

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
        self._LOGGER.info(
            "Overall communication volume: sent {} bytes, received {} bytes.".
            format(self.send_volume_intotal,
                   self.recv_volume_intotal))
        if dist.is_initialized():
            dist.destroy_process_group()

    def send(self, content=None, message_code=None, dst=0, count=True):
        """Send tensor to process rank=dst"""
        pack = Package(message_code=message_code, content=content)
        PackageProcessor.send_package(pack, dst=dst)
        if pack.content is not None and count is True:
            self.send_volume_intotal += pack.content.numel() * type2byte[
                pack.dtype]

        self._LOGGER.info(
            "Sent package to destination {}, message code {}, content length {}"
            .format(dst, message_code,
                    0 if pack.content is None else pack.content.numel()))

    def recv(self, src=None, count=True):
        """Receive tensor from process rank=src"""
        sender_rank, message_code, content = PackageProcessor.recv_package(
            src=src)

        if content is not None and count is True:
            volumn = sum([data.numel() for data in content])

            # content from server to client, the first content is id_list.
            # remove the size of id_list in the count.
            if self.rank != 0:
                volumn -= content[0].numel()

            self.recv_volume_intotal += volumn * type2byte[content[0].dtype]

        self._LOGGER.info(
            "Received package from source {}, message code {}, content length {}"
            .format(sender_rank, message_code,
                    0 if content is None else volumn))
        return sender_rank, message_code, content

    def __str__(self):
        info_str = "torch.distributed connection is initializing with ip address {}:{}, rank {}, world size: {}, backend {}, ethernet {}.".format(
            self.address[0],
            self.address[1],
            self.rank,
            self.world_size,
            self.dist_backend,
            self.ethernet,
        )
        return info_str
