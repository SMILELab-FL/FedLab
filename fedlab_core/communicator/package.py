import torch
import torch.distributed as dist
from fedlab_utils.serialization import SerializationTool
from fedlab_utils.message_code import MessageCode


HEADER_SENDER_RANK_IDX = 0
HEADER_RECEIVER_RANK_IDX = 1
HEADER_CONTENT_SIZE_IDX = 2
HEADER_MESSAGE_CODE_IDX = 3

DEFAULT_RECEIVER_RANK = -1
DEFAULT_CONTENT_SIZE = 0
DEFAULT_MESSAGE_CODE_VALUE = -1

HEADER_SIZE = 4


class Package(object):
    """A basic network package data structure used in FedLab. Everything is Tensor in FedLab.

    :class:`Package` maintains 2 variables:
        :attr:`header` : ``torch.Tensor([sender_rank, recv_rank, content_size, message_code])``
        :attr:`content` : ``torch.Tensor([offset_1, tensor_1, offset_2, tensor_2, ...])``

    Args:
        message_code (MessageCode): Agreements code defined in :class:`MessageCode`
        header (list, optional): A list containing 4 elements representing sender rank (int), receiver rank (int),
    content size (int), message code (:class:`MessageCode`) respectively.
        content (torch.Tensor, optional): Details shows above.
    """
    def __init__(self, receiver_rank=None, message_code=None, content=None):
        if receiver_rank is None:
            receiver_rank = DEFAULT_RECEIVER_RANK

        assert isinstance(receiver_rank,
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
        self.header[HEADER_SENDER_RANK_IDX] = dist.get_rank()
        self.header[HEADER_RECEIVER_RANK_IDX] = receiver_rank
        self.header[HEADER_MESSAGE_CODE_IDX] = message_code
        self.header[HEADER_CONTENT_SIZE_IDX] = DEFAULT_CONTENT_SIZE

        # initialize content
        self.content_flag = False
        self.content = torch.zeros(size=(1, ))
        if content is not None:
            if isinstance(content, torch.Tensor):
                content = [content]
            self.append_tensor_list(content)

    def append_tensor(self, tensor):
        """Append new tensor to :attr:`Package.content`
            
        Args:
            tensor (torch.Tensor): Tensor to append.
        """
        offset = tensor.shape[0]
        if self.content_flag is False:
            self.content[0] = offset
            self.content = torch.cat((self.content, tensor))
            self.content_flag = True
        else:
            self.content = torch.cat(
                (self.content, torch.Tensor([offset]), tensor))

        self.header[HEADER_CONTENT_SIZE_IDX] = self.content.shape[0]

    def append_tensor_list(self, tensor_list):
        """Append a list of tensors to :attr:`Package.content`.

        Args:
            tensor_list (list[torch.Tensor]): a list of tensors to append to :attr:`Package.content`
        """
        for tensor in tensor_list:
            self.append_tensor(tensor)

    @staticmethod
    def parse_content(content):
        """Parse package content into a list of tensors

        Args:
            content (torch.Tensor): :attr:`Package.content`, a 1-D tensor composed of several 1-D tensors and their
        corresponding offsets. For more details about :class:`Package`, refer TODO: Package design

        Returns:
            [torch.Tensor]: a list of 1-D tensors parsed from ``content``
        """
        index = 0
        parse_result = []
        if content.shape[0] >= 2:
            while index < content.shape[0]:
                offset = int(content[index])
                index += 1
                segment = content[index:index + offset]
                parse_result.append(segment)
                index += offset
        return parse_result

    @staticmethod
    def parse_header(header):
        """Parse header to get information of current package

        Args:
            header (torch.Tensor): :attr:`Package.header`, a 1-D tensor composed of 4 elements:
        ``torch.Tensor([sender_rank, recv_rank, content_size, message_code])``. For more details about :class:`Package`,
        refer TODO: Package design

        Returns:
            tuple: a tuple containing 4 elements ``(sender_rank, recv_rank, content_size, message_code)``
        """
        sender_rank = int(header[HEADER_SENDER_RANK_IDX])
        receiver_rank = int(header[HEADER_RECEIVER_RANK_IDX])
        content_size = int(header[HEADER_CONTENT_SIZE_IDX])
        message_code = MessageCode(int(header[HEADER_MESSAGE_CODE_IDX]))
        
        return sender_rank, receiver_rank, content_size, message_code
