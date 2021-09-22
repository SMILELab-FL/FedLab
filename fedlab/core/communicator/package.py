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



from typing import List
from copy import deepcopy
import torch
import torch.distributed as dist
from ...utils.message_code import MessageCode

from . import HEADER_SENDER_RANK_IDX, HEADER_RECEIVER_RANK_IDX, HEADER_SLICE_SIZE_IDX, HEADER_MESSAGE_CODE_IDX, HEADER_DATA_TYPE_IDX
from . import DEFAULT_RECEIVER_RANK, DEFAULT_SLICE_SIZE, DEFAULT_MESSAGE_CODE_VALUE
from . import HEADER_SIZE
from . import DATA_TYPE_FLOAT, DATA_TYPE_INT

class Package(object):
    """A basic network package data structure used in FedLab. Everything is Tensor in  FedLab.

    :class:`Package` maintains 3 variables:
        :attr:`header` : ``torch.Tensor([sender_rank, recv_rank, content_size, message_code, data_type])``
        :attr:`slices` : ``list[slice_size_1, slice_size_2]``
        :attr:`content` : ``torch.Tensor([tensor_1, tensor_2, ...])``

    Note:
        ``slice_size_i = tensor_i.shape[0]``, that is, every element in slices indicates the size
        of a sub-Tensor in content.

    Args:
        receiver_rank (int, optional): Rank of receiver
        message_code (MessageCode): Message code
        content (torch.Tensor, optional): Tensors contained in this package.
        data_type (int): 0 for float, 1 for int.
    """
    def __init__(self,
                 receiver_rank=None,
                 message_code=None,
                 content=None,
                 data_type=DATA_TYPE_FLOAT):
        if receiver_rank is None:
            receiver_rank = DEFAULT_RECEIVER_RANK

        assert isinstance(
            receiver_rank,
            int), "receiver_rank should be integer, not {}".format(
                type(receiver_rank))

        if message_code is None:
            message_code = DEFAULT_MESSAGE_CODE_VALUE
        else:
            if isinstance(message_code, MessageCode):
                message_code = message_code.value
        assert isinstance(
            message_code, int
        ), "message_code can only be MessageCode or integer, not {}".format(
            type(message_code))

        # initialize header
        self.header = torch.Tensor(size=(HEADER_SIZE, ))
        if dist.is_initialized():
            self.header[HEADER_SENDER_RANK_IDX] = dist.get_rank()
        else:
            self.header[HEADER_SENDER_RANK_IDX] = -1

        self.header[HEADER_RECEIVER_RANK_IDX] = receiver_rank
        self.header[HEADER_MESSAGE_CODE_IDX] = message_code
        self.header[HEADER_SLICE_SIZE_IDX] = DEFAULT_SLICE_SIZE

        if data_type == DATA_TYPE_INT:
            self.header[HEADER_DATA_TYPE_IDX] = DATA_TYPE_INT
        else:
            self.header[HEADER_DATA_TYPE_IDX] = DATA_TYPE_FLOAT

        # initialize content and slices
        self.slices = []
        self.content = None

        if isinstance(content, torch.Tensor):
            self.append_tensor(content)
        if isinstance(content, List):
            self.append_tensor_list(content)

    def append_tensor(self, tensor):
        """Append new tensor to :attr:`Package.content`

        Args:
            tensor (torch.Tensor): Tensor to append in content.
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Invalid content type")
        if tensor.shape != tensor.view(-1).shape:
            raise ValueError("Invalid shape")

        size = tensor.shape[0]
        if self.content is None:
            self.content = deepcopy(tensor)
        else:
            self.content = torch.cat((self.content, tensor))

        self.slices.append(size)
        self.header[HEADER_SLICE_SIZE_IDX] = len(self.slices)

    def append_tensor_list(self, tensor_list):
        """Append a list of tensors to :attr:`Package.content`.

        Args:
            tensor_list (list[torch.Tensor]): A list of tensors to append to :attr:`Package.content`.
        """
        for tensor in tensor_list:
            self.append_tensor(tensor)

    @staticmethod
    def parse_content(slices, content):
        """Parse package content into a list of tensors

        Args:
            slices (list[int]): A list containing number of elements of each tensor. Each number is used as offset in parsing process.
            content (torch.Tensor): :attr:`Package.content`, a 1-D tensor composed of several 1-D tensors and their
        corresponding offsets. For more details about :class:`Package`.

        Returns:
            [torch.Tensor]: a list of 1-D tensors parsed from ``content``
        """
        index = 0
        parse_result = []
        for offset in slices:
            seg_tensor = content[index:index + offset]
            parse_result.append(seg_tensor)
            index += offset
        return parse_result

    @staticmethod
    def parse_header(header):
        """Parse header to get information of current package

        Args:
            header (torch.Tensor): :attr:`Package.header`, a 1-D tensor composed of 4 elements: ``torch.Tensor([sender_rank, recv_rank, slice_size, message_code, data_type])``. For more details about :class:`Package`.

        Returns:
            tuple: A tuple containing 5 elements: ``(sender_rank, recv_rank, slice_size, message_code, data_type)``.
        """
        sender_rank = int(header[HEADER_SENDER_RANK_IDX])
        receiver_rank = int(header[HEADER_RECEIVER_RANK_IDX])
        slice_size = int(header[HEADER_SLICE_SIZE_IDX])
        message_code = MessageCode(int(header[HEADER_MESSAGE_CODE_IDX]))
        data_type = int(header[HEADER_DATA_TYPE_IDX])

        return sender_rank, receiver_rank, slice_size, message_code, data_type
