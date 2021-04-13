import torch
import torch.distributed as dist
from fedlab_core.utils.serialization import SerializationTool
from fedlab_core.utils.message_code import MessageCode


class MessageProcessor(object):
    def __init__(self, control_code_size, model) -> None:
        """
        if u want to define communication agreements, override `pack` and `unpack`
        """
        serialized_parameters = SerializationTool.serialize_model(model=model)
        self.id_size = 1
        self.control_codes_size = control_code_size
        self.serialized_param_size = serialized_parameters.numel()
        self.msg_cache = torch.zeros(
            size=(self.id_size + self.control_codes_size + self.serialized_param_size,)).cpu()

    def send_package(self, payload, dst):
        # send package
        dist.send(tensor=payload, dst=dst)

    def recv_package(self, src=None):
        # receive package
        dist.recv(tensor=self.msg_cache, src=src)
        return self.msg_cache

    def pack(self, control_codes, model):
        """
        Args:
            control_codes (list): a list of integer numbers, with each integer as control code
            model (torch.nn.Module)
        """
        # pack up Tensor
        payload = torch.Tensor([dist.get_rank()] + control_codes).cpu()
        if model is not None:
            payload = torch.cat(
                (payload, SerializationTool.serialize_model(model)))
        return payload

    def unpack(self, payload):
        sender = int(payload[self.id_size - 1])  # id_size=1 as default
        control_codes = MessageCode(int(payload[1]))
        serialized_parameters = payload[self.id_size + self.control_codes_size:]

        # TODO:  control_codes_size = 1 as default, so only return the first control code
        return sender, control_codes[0], serialized_parameters
