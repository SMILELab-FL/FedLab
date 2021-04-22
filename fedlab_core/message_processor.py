import torch
import torch.distributed as dist
from fedlab_core.utils.serialization import SerializationTool
from fedlab_core.utils.message_code import MessageCode


class Package(object):
    """
    """
    def __init__(self, header, model) -> None:
        self.header = header
        self.content = SerializationTool.serialize_model(model)

    def pack(self, header, model):
        """
        Args:
            header (list): a list of numbers(int/float), and the meaning of each number should be define in unpack
            model (torch.nn.Module)
        """
        # pack up Tensor
        header = torch.Tensor([dist.get_rank()] + header).cpu()
        if model is not None:
            payload = torch.cat(
                (header, SerializationTool.serialize_model(model)))
        return payload


class MessageProcessor(object):
    def __init__(self, header_size, model) -> None:
        """Define the details of how the topology module to deal with network communication
        if u want to define communication agreements, override `pack` and `unpack`

        `class: MessageProcessor` will create message cache according to args
        args:
            header_size (int): Size of header
            model (torch.nn.Module): Model used in federation
        """
        serialized_parameters = SerializationTool.serialize_model(model=model)
        self.header_size = max(1, header_size+1)
        self.serialized_param_size = serialized_parameters.numel()
        self.msg_cache = torch.zeros(
            size=(self.header_size + self.serialized_param_size,)).cpu()

    def send_package(self, payload, dst):
        # send package
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
        #TODO: when model is none
        # pack up Tensor
        header = torch.Tensor([dist.get_rank()] + header).cpu()
        if model is not None:
            payload = torch.cat(
                (header, SerializationTool.serialize_model(model)))
        return payload

    def unpack(self, payload):
        sender = int(payload[0])
        header = MessageCode(int(payload[1]))
        serialized_parameters = payload[self.header_size:]
        return sender, header, serialized_parameters
