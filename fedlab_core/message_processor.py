import torch
import torch.distributed as dist
from fedlab_core.utils.serialization import SerializationTool
from fedlab_core.utils.message_code import MessageCode

SENDER_IDX = 0
RECVER_IDX = 1
CONTENTSIZE_IDX = 2
MESSAGECODE_IDX = 3

DEFAULT_RANK = -1
DEFAULT_CS = -1
DEFAULT_MC = -1

HEADER_SIZE = 4


class Package(object):
    """
    FedLab网络包的基本单位
    由两部分序列tensor构成
        header : [sender_rank, recver_rank, content_size, message_code]
        content : [[offset,info]]
    """

    def __init__(self, recver_rank=None, message_code=None) -> None:
        self.header = [dist.get_rank(), DEFAULT_RANK, DEFAULT_CS,
                       DEFAULT_MC]  # header固定4位
        self.content = torch.zeros(size=(1,))
        self.content_flag = False
        self.header_flag = False

        if message_code is not None:
            self.header[MESSAGECODE_IDX] = message_code.value

        if recver_rank is not None:
            self.header[RECVER_IDX] = recver_rank

    def append_tensor(self, tensor):
        if self.content_flag is False:
            self.content[0] = tensor.shape[0]
            self.content = torch.cat((self.content, tensor))
            self.content_flag = True
        else:
            offset = tensor.shape[0]
            self.content = torch.cat(
                (self.content, torch.Tensor([offset]), tensor))

        self.header[CONTENTSIZE_IDX] = self.content.shape[0]

    def append_tensor_list(self, tensor_list):
        for tensor in tensor_list:
            offset = tensor.shape[0]
            cache_tensor = torch.cat((torch.Tensor([offset]), tensor))
            self.append(cache_tensor)

        self.header[CONTENTSIZE_IDX] = self.content.shape[0]

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
            segment = content[index:index+offset]
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
            return int(cache[SENDER_IDX]), int(cache[RECVER_IDX]), int(cache[CONTENTSIZE_IDX]), MessageCode(int(cache[MESSAGECODE_IDX]))

        def recv_content(cache_size, src):
            cache = torch.zeros(size=(cache_size,))
            dist.recv(cache, src=src)
            return Package.unpack_content(cache)

        sender, recver, content_size, message_code = recv_header(src=src)

        # 收到第一段包，第二段包指定来源rank
        content = recv_content(content_size, src=sender)

        return message_code, sender, content

    @staticmethod
    def send_package(package, dst):
        def send_header(header, dst):
            header = torch.cat((torch.Tensor([dist.get_rank(), dst]), header))
            dist.send(header, dst=dst)

        def send_content(content, dst):
            dist.send(content, dst=dst)

        header = package.header
        header[RECVER_IDX] = dst        # 接收者rank写入

        send_header(header, dst=dst)

        send_content(content=package.content, dst=dst)


class MessageProcessor(object):
    """Define the details of how the topology module to deal with network communication
    if u want to define communication agreements, override `pack` and `unpack`

    `class: MessageProcessor` will create message cache according to args
    args:
        header_instance (int): a instance of header (rank of sender and recver is not included)
        model (torch.nn.Module): Model used in federation
    """
    @staticmethod
    def send_package(model, message_code, dst):
        # send package
        s_parameters = SerializationTool.serialize_model(model)
        content_size = s_parameters.shape[0]
        package = torch.cat((torch.Tensor([dist.get_rank(), int(dst), content_size, message_code]), s_parameters))
        dist.send(tensor=package, dst=dst)

    @staticmethod
    def recv_package(model, src=None):
        def unpack(payload):
            sender = int(payload[SENDER_IDX])
            recver = int(payload[RECVER_IDX])
            content_size = int(payload[CONTENTSIZE_IDX])
            
            message_code = MessageCode(int(payload[MESSAGECODE_IDX]))
            serialized_parameters = payload[HEADER_SIZE:]

            return sender, recver, content_size, message_code, serialized_parameters

        s_parameters = SerializationTool.serialize_model(model)
        pkg_cache = torch.zeros(size=(HEADER_SIZE + s_parameters.shape[0],)).cpu()
        dist.recv(tensor=pkg_cache, src=src)

        sender, _, _, message_code, s_parameters = unpack(pkg_cache)
        return sender, message_code, s_parameters
