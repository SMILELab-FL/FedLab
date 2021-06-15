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

    """
    @property
    def header(self):
        return self.header

    @property
    def content(self):
        return self.content
    """

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


class PackageProcessor(object):
    """Provide more flexible distributed tensor communication functions based on :func:`torch.distributed.send` and
    :func:`torch.distributed.recv`"""
    @staticmethod
    def recv_package(src=None):
        def recv_header(src=src, parse=True):
            buffer = torch.zeros(size=(HEADER_SIZE, ))
            dist.recv(buffer, src=src)
            if parse is True:
                return Package.parse_header(buffer)
            else:
                return buffer

        def recv_content(cache_size, src):
            buffer = torch.zeros(size=(cache_size, ))
            dist.recv(buffer, src=src)
            return Package.parse_content(buffer)

        sender_rank, recv_rank, content_size, message_code = recv_header(
            src=src)

        # 收到第一段包，第二段包指定来源rank
        if content_size > 0:
            content = recv_content(content_size, src=sender_rank)
        else:
            content = None

        return sender_rank, message_code, content

    @staticmethod
    def send_package(package, dst):
        """Two-segment tensor communication pattern based on ``torch.distributed``

        Pattern is shown as follows:
            1.1 sender: send a header tensor containing ``content_size`` to receiver
            1.2 receiver: receive the header, and get the value of ``content_size`` and create a buffer for incoming content

            2.1 sender: send a content tensor composed of a list of tensors and their offsets
            2.2 receiver: receive the content tensor, and parse it to obtain a tensor list using parser function
        """
        def send_header(header, dst):
            header[HEADER_RECEIVER_RANK_IDX] = dst
            dist.send(header, dst=dst)

        def send_content(content, dst):
            dist.send(content, dst=dst)

        send_header(header=package.header, dst=dst)

        if package.header[HEADER_CONTENT_SIZE_IDX] > 0:
            send_content(content=package.content, dst=dst)


"""
    # 暂时弃用
    @staticmethod
    def send_model(model, message_code, dst):
        def pack(dst, content_size, message_code, s_params):
            return torch.cat((torch.Tensor(
                [dist.get_rank(), dst, content_size, message_code]), s_params))

        # send package
        if isinstance(torch.nn.Module):
            serialized_params = SerializationTool.serialize_model(model)
        else:
            serialized_params = model

        content_size = serialized_params.shape[0]
        package = pack(dst, content_size, message_code, serialized_params)
        dist.send(tensor=package, dst=dst)

    @staticmethod
    def recv_model(model, src=None):
        def unpack(package):
            sender_rank = int(package[HEADER_SENDER_RANK_IDX])
            recv_rank = int(package[HEADER_RECEIVER_RANK_IDX])
            content_size = int(package[HEADER_CONTENT_SIZE_IDX])

            message_code = MessageCode(int(package[HEADER_MESSAGE_CODE_IDX]))
            s_params = package[HEADER_SIZE:]

            return sender_rank, recv_rank, content_size, message_code, s_params

        serialized_params = SerializationTool.serialize_model(model)
        pkg_cache = torch.zeros(size=(HEADER_SIZE +
                                      serialized_params.shape[0], )).cpu()
        dist.recv(tensor=pkg_cache, src=src)

        sender_rank, _, _, message_code, serialized_params = unpack(pkg_cache)
        return sender_rank, message_code, serialized_params
"""