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

import warnings
from typing import List
from copy import deepcopy
import torch
import torch.distributed as dist

from . import HEADER_SENDER_RANK_IDX, HEADER_RECEIVER_RANK_IDX, HEADER_SLICE_SIZE_IDX, \
    HEADER_MESSAGE_CODE_IDX, HEADER_DATA_TYPE_IDX
from . import DEFAULT_SLICE_SIZE, DEFAULT_MESSAGE_CODE_VALUE
from . import HEADER_SIZE
from ...utils.message_code import MessageCode

supported_torch_dtypes = [
    torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
    torch.float32, torch.float64
]


class Package(object):
    """A basic network package data structure used in FedLab. Everything is Tensor in  FedLab.

    Note:
        ``slice_size_i = tensor_i.shape[0]``, that is, every element in slices indicates the size
        of a sub-Tensor in content.

    :class:`Package` maintains 3 variables:
        - :attr:`header` : ``torch.Tensor([sender_rank, recv_rank, content_size, message_code, data_type])``
        - :attr:`slices` : ``list[slice_size_1, slice_size_2]``
        - :attr:`content` : ``torch.Tensor([tensor_1, tensor_2, ...])``

    Args:
        message_code (MessageCode): Message code
        content (torch.Tensor, optional): Tensors contained in this package.
    """

    def __init__(self, message_code=None, content=None):

        if message_code is None:
            message_code = DEFAULT_MESSAGE_CODE_VALUE
        else:
            if isinstance(message_code, MessageCode):
                message_code = message_code.value
        assert isinstance(
            message_code, int
        ), "message_code can only be MessageCode or integer, not {}".format(
            type(message_code))

        # initialize header. The dtype of header is set as torch.int32 as default.
        self.header = torch.zeros(size=(HEADER_SIZE, ), dtype=torch.int32)

        if dist.is_initialized():
            self.header[HEADER_SENDER_RANK_IDX] = dist.get_rank()
        else:
            self.header[HEADER_SENDER_RANK_IDX] = -1
        self.header[HEADER_RECEIVER_RANK_IDX] = -1  # assigned by processor
        self.header[HEADER_MESSAGE_CODE_IDX] = message_code
        self.header[HEADER_SLICE_SIZE_IDX] = DEFAULT_SLICE_SIZE
        self.header[HEADER_DATA_TYPE_IDX] = -1  # assigned by processor

        # initialize content and slices
        self.slices = []
        self.content = None
        self.dtype = None

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
            raise ValueError(
                "Invalid content type, expecting torch.Tensor but get {}".
                format(type(tensor)))

        shape = list(tensor.shape)
        slice = [tensor.numel(), len(shape)] + shape

        tensor = tensor.view(-1)
        if self.content is None:
            self.content = deepcopy(tensor)
            self.dtype = tensor.dtype
        else:
            if tensor.dtype is not self.dtype:
                warnings.warn(
                    "The dtype of current tensor is {}. But package dtype is {}. The current data type will be casted to {} and fedlab do not guarantee lossless conversion."
                    .format(tensor.dtype, self.dtype, self.dtype))
            tensor = tensor.to(self.dtype)
            self.content = torch.cat((self.content, tensor))

        self.slices += slice
        self.header[HEADER_SLICE_SIZE_IDX] = len(self.slices)

    def append_tensor_list(self, tensor_list):
        """Append a list of tensors to :attr:`Package.content`.

        Args:
            tensor_list (list[torch.Tensor]): A list of tensors to append to :attr:`Package.content`.
        """
        for tensor in tensor_list:
            self.append_tensor(tensor)

    def to(self, dtype):
        if dtype in supported_torch_dtypes:
            self.dtype = dtype
            self.content.to(self.dtype)
        else:
            warnings.warn(
                "FedLab only supports following data types: torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64."
            )

    @staticmethod
    def parse_content(slices, content):
        """Parse package content into a list of tensors

        Args:
            slices (list[int]): A list containing number of elements of each tensor. Each number is used as offset in parsing process.
            content (torch.Tensor): :attr:`Package.content`, a 1-D tensor composed of several 1-D tensors and their corresponding offsets. For more details about :class:`Package`.

        Returns:
            list[torch.Tensor]: A list of 1-D tensors parsed from ``content``
        """
        index = 0  # parse variable for content
        iter = 0  # parse variable for slices
        parse_result = []
        while iter < len(slices):
            offset = slices[iter]  # offset of content
            shape_len = slices[iter + 1]  # offset of shape tuple
            shape = tuple(slices[iter + 2:iter + 2 +
                                 shape_len])  # obtain shape tuple

            seg_tensor = content[index:index + offset]
            reshape_tensor = seg_tensor.view(size=shape)  # reshape

            parse_result.append(reshape_tensor)
            index += offset
            iter += shape_len + 2

        return parse_result

    @staticmethod
    def parse_header(header):
        """Parse header to get information of current package.

        Args:
            header (torch.Tensor): :attr:`Package.header`, a 1-D tensor composed of 4 elements: ``torch.Tensor([sender_rank, recv_rank, slice_size, message_code, data_type])``.
            For more details about :class:`Package`.

        Returns:
            tuple: A tuple containing 5 elements: ``(sender_rank, recv_rank, slice_size, message_code, data_type)``.
        """
        sender_rank = int(header[HEADER_SENDER_RANK_IDX])
        receiver_rank = int(header[HEADER_RECEIVER_RANK_IDX])
        slice_size = int(header[HEADER_SLICE_SIZE_IDX])
        message_code = MessageCode(int(header[HEADER_MESSAGE_CODE_IDX]))
        data_type = int(header[HEADER_DATA_TYPE_IDX])

        return sender_rank, receiver_rank, slice_size, message_code, data_type
