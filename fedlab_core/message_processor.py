from enum import Enum
import torch
import torch.distributed as dist


class MessageCode(Enum):
    """Different types of messages between client and server that we support go here."""
    ParameterRequest = 0
    GradientUpdate = 1
    ParameterUpdate = 2
    EvaluateParams = 3
    Exit = 4


class SerializationTool(object):
    @staticmethod
    def serlialize_model(model):
        """
        vectorize each model parameter
        """
        parameters = [param.data.view(-1) for param in model.parameters()]
        m_parameters = torch.cat(parameters)
        m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def resotre_model(model, serialized_parameters):
        """
        Assigns grad_update params to model.parameters.
        This is done by iterating through `model.parameters()` and assigning the relevant params in `grad_update`.
        NOTE: this function manipulates `model.parameters`.
        """
        current_index = 0  # keep track of where to read from grad_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            parameter.data.copy_(
                serialized_parameters[current_index:current_index + numel].view(size))
            current_index += numel


class MessageProcessor(object):
    def __init__(self, control_code_size, model) -> None:
        """
        if u want to define communication agreements, override `pack` and `unpack`
        """
        # TODO: add assert
        serialized_parameters = SerializationTool.serlialize_model(model=model)
        self.cc_size = control_code_size
        self.sp_size = serialized_parameters.numel()
        self.msg_cache = torch.zeros(size=(self.cc_size+self.sp_size,)).cpu()

        class payload(object):
            def __init__(self, control_codes, model) -> None:
                super().__init__()

    def send_package(self, payload, dst):
        # send package
        dist.send(tensor=payload, dst=dst)

    def recv_package(self, src=None):
        # receive package
        dist.recv(tensor=self.msg_cache, src=src)
        return self.msg_cache

    def pack(self, control_codes, model):
        # pack up Tensor
        payload = torch.Tensor([dist.get_rank()] + control_codes).cpu()
        if model is not None:
            payload = torch.cat(
                (payload, SerializationTool.serlialize_model(model)))
        return payload

    def unpack(self, payload):
        sender = int(payload[0])
        message_code = MessageCode(int(payload[1]))
        s_parameters = payload[2:]

        return sender, message_code, s_parameters
