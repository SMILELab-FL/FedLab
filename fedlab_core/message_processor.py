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
    """FedLab网络包的基本单位

    由两部分序列tensor构成
        header : [sender_rank, recv_rank, content_size, message_code]
        content : [offset_1, tensor_1, offset_2, tensor_2, ...]
    """

    def __init__(self, message_code) -> None:
        self.header = [dist.get_rank(), DEFAULT_RECV_RANK, DEFAULT_CONTENT_SIZE,
                       message_code]
        self.content = torch.zeros(size=(1,))
        self.content_flag = False

    def append_tensor(self, tensor):
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
        index = 0
        parse_result = []
        while index < content.shape[0]:
            offset = int(content[index])
            index += 1
            segment = content[index:index + offset]
            parse_result.append(segment)
            index += offset
        return parse_result


class PackageProcessor(object):
    # TODO： 添加包内容合法检测
    @staticmethod
    def recv_package(src=None):
        def recv_header(src=src):
            cache = torch.zeros(size=(HEADER_SIZE,))
            dist.recv(cache, src=src)
            return int(cache[HEADER_SENDER_RANK_IDX]), int(cache[HEADER_RECVER_RANK_IDX]), int(
                cache[HEADER_CONTENT_SIZE_IDX]), MessageCode(
                int(cache[HEADER_MESSAGE_CODE_IDX]))

        def recv_content(cache_size, src):
            cache = torch.zeros(size=(cache_size,))
            dist.recv(cache, src=src)
            return Package.parse_content(cache)

        sender_rank, recv_rank, content_size, message_code = recv_header(src=src)

        # 收到第一段包，第二段包指定来源rank
        content = recv_content(content_size, src=sender_rank)

        return message_code, sender_rank, content

    @staticmethod
    def send_package(package, dst):
        def send_header(header, dst):
            header[HEADER_RECVER_RANK_IDX] = dst
            dist.send(header, dst=dst)

        def send_content(content, dst):
            dist.send(content, dst=dst)

        send_header(header=package.header, dst=dst)
        send_content(content=package.content, dst=dst)


class MessageProcessor(object):
    """Define the details of how the topology module to deal with network communication
    if u want to define communication agreements, override :func:`pack` and :func:`unpack`

    :class:`MessageProcessor` will create message cache according to args.

    # Args:
        # header_instance (int): a instance of header (rank of sender and recv is not included)
        # model (torch.nn.Module): Model used in federation
    """

    @staticmethod
    def send_package(model, message_code, dst):
        def pack(dst, content_size, message_code, s_params):
            return torch.cat((torch.Tensor([dist.get_rank(), dst, content_size, message_code]), s_params))

        # send package
        serialized_params = SerializationTool.serialize_model(model)
        content_size = serialized_params.shape[0]
        package = pack(dst, content_size, message_code, serialized_params)
        dist.send(tensor=package, dst=dst)

    @staticmethod
    def recv_package(model, src=None):
        def unpack(package):
            sender_rank = int(package[HEADER_SENDER_RANK_IDX])
            recv_rank = int(package[HEADER_RECVER_RANK_IDX])
            content_size = int(package[HEADER_CONTENT_SIZE_IDX])

            message_code = MessageCode(int(package[HEADER_MESSAGE_CODE_IDX]))
            s_params = package[HEADER_SIZE:]

            return sender_rank, recv_rank, content_size, message_code, s_params

        serialized_params = SerializationTool.serialize_model(model)
        pkg_cache = torch.zeros(size=(HEADER_SIZE + serialized_params.shape[0],)).cpu()
        dist.recv(tensor=pkg_cache, src=src)

        sender, _, _, message_code, serialized_params = unpack(pkg_cache)
        return sender, message_code, serialized_params
