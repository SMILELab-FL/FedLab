from typing import List
import torch
import torch.distributed as dist
from fedlab_utils.message_code import MessageCode
from copy import deepcopy

HEADER_SENDER_RANK_IDX = 0
HEADER_RECEIVER_RANK_IDX = 1
HEADER_SLICE_SIZE_IDX = 2
HEADER_MESSAGE_CODE_IDX = 3

DEFAULT_RECEIVER_RANK = -1
DEFAULT_SLICE_SIZE = 0
DEFAULT_MESSAGE_CODE_VALUE = -1

HEADER_SIZE = 4

class Package(object):
    """A basic network package data structure used in FedLab. Everything is Tensor in  FedLab.

    :class:`Package` maintains 3 variables:
        :attr:`header` : ``torch.Tensor([sender_rank, recv_rank, content_size, message_code])``
        :attr:`slices` : ``list[slice_size_1, slice_size_2]``
        :attr:`content` : ``torch.Tensor([tensor_1, tensor_2, ...])``

    Note:
        slice_size_i = tensor_i.shape[0]
        every element in slices indicates the size of a sub-Tensor in content.

    Args:
        receiver_rank (int, optional): rank of receiver
        message_code (message code): message code
        content (torch.Tensor, optional): Details shows above.
    """
    def __init__(self, receiver_rank=None, message_code=None, content=None):
        if receiver_rank is None:
            receiver_rank = DEFAULT_RECEIVER_RANK

        assert isinstance(
            receiver_rank,
            int), 'receiver_rank should be integer, not {}'.format(
                type(receiver_rank))

        if message_code is None:
            message_code = DEFAULT_MESSAGE_CODE_VALUE
        else:
            if isinstance(message_code, MessageCode):
                message_code = message_code.value
        assert isinstance(
            message_code, int
        ), 'message_code can only be MessageCode or integer, not {}'.format(
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
            tensor_list (list[torch.Tensor]): a list of tensors to append to :attr:`Package.content`
        """
        for tensor in tensor_list:
            self.append_tensor(tensor)

    @staticmethod
    def parse_content(slices, content):
        """Parse package content into a list of tensors

        Args:
            slices (list): 
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
            header (torch.Tensor): :attr:`Package.header`, a 1-D tensor composed of 4 elements:
        ``torch.Tensor([sender_rank, recv_rank, content_size, message_code])``. For more details about :class:`Package`.

        Returns:
            tuple: a tuple containing 4 elements ``(sender_rank, recv_rank, slice_size, message_code)``
        """
        sender_rank = int(header[HEADER_SENDER_RANK_IDX])
        receiver_rank = int(header[HEADER_RECEIVER_RANK_IDX])
        slice_size = int(header[HEADER_SLICE_SIZE_IDX])
        message_code = MessageCode(int(header[HEADER_MESSAGE_CODE_IDX]))

        return sender_rank, receiver_rank, slice_size, message_code
