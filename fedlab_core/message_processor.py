from enum import Enum

import torch
from torch import serialization
import torch.distributed as dist

from fedlab_core.utils.serialization import ravel_model_params


class MessageCode(Enum):
    """Different types of messages between client and server that we support go here."""
    ParameterRequest = 0
    GradientUpdate = 1
    ParameterUpdate = 2
    EvaluateParams = 3
    Exit = 4


class MessageProcessor(object):
    """

    """

    def __init__(self, control_code_size, model) -> None:
        # TODO: add assert
        serialized_parameters = self.ravel_model_params(model)
        self.cc_size = control_code_size
        self.sp_size = serialized_parameters.shape[0]
        self.msg_cache = torch.Tensor(shape=(self.cc_size+self.sp_size))

    def send_message(self, control_codes, s_parameters, dst):
        """Sends a message to destination.

        Concatenates destination rank, message code and payload into a single tensor, then sends it.

        Args:
            s_parameters (tensor): Serialized model parameters
            control_codes (tensor or int/float): Control message defined by user and its shape should be (1,)
            dst (int, optional): Destination rank. Default is 0 which means the server

        Returns:
            Serialized message package: [current process rank, control codes, serialized model parameters]
        """
        message = torch.cat(
            (torch.Tensor([dist.get_rank()]), torch.tensor(control_codes)))

        # message is used in communication, therefor use .cpu()
        message = torch.cat((message, s_parameters.cpu()))
        dist.send(tensor=message, dst=dst)

        return message

    def recv_message(self, src=0):
        """Receives message from source.

        Args:
            sp_size (int): the size of serialized model parameters. This varies help function to create correct size of cache tensor
            control_code_size (int): the size of control code. This varies help function to create correct size of cache tensor
            src (int, optional): Source rank. Will receive from any process if unspecified.

        Returns:
            (rank of sender, list of contro codes, serialized model parameters)
        """
        print("RECV MESSAGE: RANK: {}".format(dist.get_rank()))  # debug

        dist.recv(tensor=self.msg_cache, src=src)

        # this function dose not manipulate model prameters!
        # return parsed information: sender rank, control codes, serialized model parameters
        return self.msg_cache[0:self.cc_size], self.msg_cache[self.cc_size:]

    def ravel_model_params(self, model, cuda=False):
        # vectorize each model parameter
        parameters = [param.data.view(-1) for param in model.parameters()]
        m_parameters = torch.cat(parameters)

        if cuda:
            m_parameters = m_parameters.cuda()

        return m_parameters

    def unravel_model_params(self, model, serialized_parameters):
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
