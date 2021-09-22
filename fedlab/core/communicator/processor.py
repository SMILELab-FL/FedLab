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

import numpy as np

import torch
import torch.distributed as dist

from .package import Package
from . import HEADER_SIZE, HEADER_RECEIVER_RANK_IDX, HEADER_SLICE_SIZE_IDX


class PackageProcessor(object):
    """Provide more flexible distributed tensor communication functions based on
    :func:`torch.distributed.send` and :func:`torch.distributed.recv`.

    Notes:
        EVERYTHING is :class:`torch.Tensor` in FedLab.
    """
    @staticmethod
    def recv_package(src=None):
        """Three-segment tensor communication pattern based on ``torch.distributed``

        Pattern is shown as follows:
            1.1 sender: send a header tensor containing ``slice_size`` to receiver

            1.2 receiver: receive the header, and get the value of ``slice_size`` and create a buffer for incoming slices of content

            2.1 sender: send a list of slices indicating the size of every content size.

            2.2 receiver: receive the slices list.

            3.1 sender: send a content tensor composed of a list of tensors.

            3.2 receiver: receive the content tensor, and parse it to obtain slices list using parser function
        """
        def recv_header(src=src, parse=True):
            buffer = torch.zeros(size=(HEADER_SIZE, ))
            dist.recv(buffer, src=src)
            if parse is True:
                return Package.parse_header(buffer)
            else:
                return buffer

        def recv_slices(slices_size, src):
            buffer_slices = torch.zeros(size=(slices_size, ),
                                        dtype=torch.int32)
            dist.recv(buffer_slices, src=src)
            slices = [x.item() for x in buffer_slices]
            return slices

        def recv_content(slices, data_type, src):
            content_size = sum(slices)
            if data_type == 0:
                buffer = torch.zeros(size=(content_size, ),
                                     dtype=torch.float32)
            else:
                buffer = torch.zeros(size=(content_size, ), dtype=torch.int32)
            dist.recv(buffer, src=src)
            return Package.parse_content(slices, buffer)

        sender_rank, _, slices_size, message_code, data_type = recv_header(
            src=src)

        if slices_size > 0:
            slices = recv_slices(slices_size=slices_size, src=sender_rank)
            content = recv_content(slices, data_type, src=sender_rank)
        else:
            content = None

        return sender_rank, message_code, content

    @staticmethod
    def send_package(package, dst):
        """Three-segment tensor communication pattern based on ``torch.distributed``

        Pattern is shown as follows:
            1.1 sender: send a header tensor containing ``slice_size`` to receiver

            1.2 receiver: receive the header, and get the value of ``slice_size`` and create a buffer for incoming slices of content

            2.1 sender: send a list of slices indicating the size of every content size.

            2.2 receiver: receive the slices list.

            3.1 sender: send a content tensor composed of a list of tensors.

            3.2 receiver: receive the content tensor, and parse it to obtain slices list using parser function
        """
        def send_header(header, dst):
            header[HEADER_RECEIVER_RANK_IDX] = dst
            dist.send(header, dst=dst)

        def send_slices(slices, dst):
            np_slices = np.array(slices, dtype=np.int32)
            tensor_slices = torch.from_numpy(np_slices)
            dist.send(tensor_slices, dst=dst)

        def send_content(content, dst):
            dist.send(content, dst=dst)

        send_header(header=package.header, dst=dst)

        if package.header[HEADER_SLICE_SIZE_IDX] > 0:
            send_slices(slices=package.slices, dst=dst)

            send_content(content=package.content, dst=dst)
