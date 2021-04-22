import torch
import torch.distributed as dist
from fedlab_core.utils.serialization import SerializationTool
from fedlab_core.utils.message_code import MessageCode

SENDER_IDX = 0
RECVER_IDX = 1


class MessageProcessor(object):
    """Define the details of how the topology module to deal with network communication
    if u want to define communication agreements, override `pack` and `unpack`

    `class: MessageProcessor` will create message cache according to args
    args:
        header_instance (int): a instance of header (rank of sender and recver is not included)
        model (torch.nn.Module): Model used in federation
    """

    def __init__(self, header_instance, model) -> None:
        serialized_parameters = SerializationTool.serialize_model(model=model)
        # TODO check type of header_instance and model
        self.header_size = max(2, len(header_instance)+2)
        self.serialized_param_size = serialized_parameters.numel()
        self.msg_cache = torch.zeros(
            size=(self.header_size + self.serialized_param_size,)).cpu()

    def send_package(self, payload, dst):
        # send package
        # 发送前添加sender->recver的rank id
        payload = torch.cat(
            (torch.Tensor([dist.get_rank(), int(dst)]), payload))
        dist.send(tensor=payload, dst=dst)

    def recv_package(self, src=None):
        # receive package
        dist.recv(tensor=self.msg_cache, src=src)
        return self.msg_cache

    def pack(self, header, model):
        """
        Args:
            header (list): a list of numbers(int/float), and the meaning of each number should be define in unpack
            model (torch.nn.Module)
        """
        payload = torch.cat(
            (torch.Tensor(header), SerializationTool.serialize_model(model)))
        return payload

    def unpack(self, payload):
        sender = int(payload[SENDER_IDX])
        recver = int(payload[RECVER_IDX])
        msg_code = MessageCode(int(payload[2]))
        serialized_parameters = payload[self.header_size:]
        return sender, recver, msg_code, serialized_parameters
