import torch
import torch.distributed as dist
from fedlab_core.utils.serialization import SerializationTool
from fedlab_core.utils.message_code import MessageCode


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
        self.id_size = 1
        self.header_size = header_size
        self.serialized_param_size = serialized_parameters.numel()
        self.msg_cache = torch.zeros(
            size=(self.id_size + self.header_size + self.serialized_param_size,)).cpu()

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
        # pack up Tensor
        payload = torch.Tensor([dist.get_rank()] + header).cpu()
        if model is not None:
            payload = torch.cat(
                (payload, SerializationTool.serialize_model(model)))
        return payload

    def unpack(self, payload):
        sender = int(payload[self.id_size - 1])  # id_size=1 as default
        header = MessageCode(int(payload[1]))
        serialized_parameters = payload[self.id_size +
                                        self.header_size:]

        # TODO:  header_size = 1 as default, so only return the first header
        return sender, header, serialized_parameters
