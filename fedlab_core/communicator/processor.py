import torch
import torch.distributed as dist
from fedlab_core.utils.serialization import SerializationTool
from fedlab_core.utils.message_code import MessageCode

HEADER_SENDER_RANK_IDX = 0
HEADER_RECVER_RANK_IDX = 1
HEADER_CONTENT_SIZE_IDX = 2
HEADER_MESSAGE_CODE_IDX = 3

DEFAULT_RECV_RANK = -1
DEFAULT_CONTENT_SIZE = -1
DEFAULT_MC = -1

HEADER_SIZE = 4


class Package(object):
    """A basic network package data structure used in FedLab. Everything is Tensor in FedLab.

        this class maintains 2 variables:
            header : torch.Tensor([sender_rank, recv_rank, content_size, message_code])
            content : torch.Tensor([offset_1, tensor_1, offset_2, tensor_2, ...])

        args:
            message_code (MessageCode): Agreements code defined in: class:`MessageCode`
            header (list, optional): Details shows above.
            content (torch.Tensor, optional): Details shows above.
    """

    def __init__(self, message_code, header=None, content=None) -> None:
        if header is not None:
            self.header = torch.Tensor([dist.get_rank(), DEFAULT_RECV_RANK, DEFAULT_CONTENT_SIZE,
                                        message_code])
        else:
            self.header = torch.Tensor(header)

        if content is not None:
            self.content = torch.zeros(size=(1,))
        else:
            self.content = content

        self.content_flag = False

    def append_tensor(self, tensor):
        """Append new tensor to content
            
            args:
                tensor (torch.Tensor): The tensor to append.
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
        """Append a list of tensors to content:

            args:
                tensor (list): a list of tensor
        """
        for tensor in tensor_list:
            self.append_tensor(tensor)

        self.header[HEADER_CONTENT_SIZE_IDX] = self.content.shape[0]

    @property
    def header(self):
        return self.header

    @property
    def content(self):
        return self.content

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
        recver_rank = int(header[HEADER_RECVER_RANK_IDX])
        content_size = int(header[HEADER_CONTENT_SIZE_IDX])
        message_code = MessageCode(int(header[HEADER_MESSAGE_CODE_IDX]))
        return sender_rank, recver_rank, content_size, message_code


class PackageProcessor(object):
    """Provide more flexible distributed tensor communication functions based on :func:`torch.distributed.send` and
    :func:`torch.distributed.recv`"""

    @staticmethod
    def recv_package(src=None):
        def recv_header(src=src, parse=False):
            cache = torch.zeros(size=(HEADER_SIZE,))
            dist.recv(cache, src=src)
            if parse is True:
                return Package.parse_header(cache)
            else:
                return cache

        def recv_content(cache_size, src):
            cache = torch.zeros(size=(cache_size,))
            dist.recv(cache, src=src)
            return Package.parse_content(cache)

        sender_rank, recv_rank, content_size, message_code = recv_header(
            src=src)

        # 收到第一段包，第二段包指定来源rank
        content = recv_content(content_size, src=sender_rank)

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
            header[HEADER_RECVER_RANK_IDX] = dst
            dist.send(header, dst=dst)

        def send_content(content, dst):
            dist.send(content, dst=dst)

        send_header(header=package.header, dst=dst)
        send_content(content=package.content, dst=dst)

    @staticmethod
    def send_model(model, message_code, dst):
        """Directely send serialized model parameters to dst"""

        def pack(dst, content_size, message_code, s_params):
            return torch.cat((torch.Tensor([dist.get_rank(), dst, content_size, message_code]), s_params))

        # send package
        serialized_params = SerializationTool.serialize_model(model)
        content_size = serialized_params.shape[0]
        package = pack(dst, content_size, message_code, serialized_params)
        dist.send(tensor=package, dst=dst)

    @staticmethod
    def recv_model(model, src=None):
        """Receive serialized model parameters from src"""

        def unpack(package):
            sender_rank = int(package[HEADER_SENDER_RANK_IDX])
            recv_rank = int(package[HEADER_RECVER_RANK_IDX])
            content_size = int(package[HEADER_CONTENT_SIZE_IDX])

            message_code = MessageCode(int(package[HEADER_MESSAGE_CODE_IDX]))
            s_params = package[HEADER_SIZE:]

            return sender_rank, recv_rank, content_size, message_code, s_params

        serialized_params = SerializationTool.serialize_model(model)
        pkg_cache = torch.zeros(
            size=(HEADER_SIZE + serialized_params.shape[0],)).cpu()
        dist.recv(tensor=pkg_cache, src=src)

        sender_rank, _, _, message_code, serialized_params = unpack(pkg_cache)
        return sender_rank, message_code, serialized_params


class MessageProcessor(object):
    """Define the details of how the topology module to deal with network communication
    if u want to define communication agreements, override :func:`pack` and :func:`unpack`

    :class:`MessageProcessor` will create message cache according to args.

    # Args:
        # header_instance (int): a instance of header (rank of sender and recv is not included)
        # model (torch.nn.Module): Model used in federation
    """

    @staticmethod
    def send_model(model, message_code, dst):
        def pack(dst, content_size, message_code, s_params):
            return torch.cat((torch.Tensor([dist.get_rank(), dst, content_size, message_code]), s_params))

        # send package
        serialized_params = SerializationTool.serialize_model(model)
        content_size = serialized_params.shape[0]
        package = pack(dst, content_size, message_code, serialized_params)
        dist.send(tensor=package, dst=dst)

    @staticmethod
    def recv_model(model, src=None):
        def unpack(package):
            sender_rank = int(package[HEADER_SENDER_RANK_IDX])
            recv_rank = int(package[HEADER_RECVER_RANK_IDX])
            content_size = int(package[HEADER_CONTENT_SIZE_IDX])

            message_code = MessageCode(int(package[HEADER_MESSAGE_CODE_IDX]))
            s_params = package[HEADER_SIZE:]

            return sender_rank, recv_rank, content_size, message_code, s_params

        serialized_params = SerializationTool.serialize_model(model)
        pkg_cache = torch.zeros(
            size=(HEADER_SIZE + serialized_params.shape[0],)).cpu()
        dist.recv(tensor=pkg_cache, src=src)

        sender_rank, _, _, message_code, serialized_params = unpack(pkg_cache)
        return sender_rank, message_code, serialized_params
