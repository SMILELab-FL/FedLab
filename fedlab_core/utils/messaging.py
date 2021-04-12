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


# 提供用户自定义控制变量的接口， 其中payload固定为序列化后的模型参数
def send_message_tmp(s_parameters, control_codes, dst):
    """Sends a message to destination.

    Concatenates destination rank, message code and payload into a single tensor, then sends it.

    Args:
        s_parameters (torch.Tensor): Serialized model parameters
        control_codes (torch.Tensor, int/float): Control message defined by user and its shape should be (1,)
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


def recv_message_tmp(sp_size, control_code_size, src=0):
    """Receives message from source.

    Args:
        sp_size (int): the size of serialized model parameters. This varies help function to create correct size of cache tensor
        control_code_size (int): the size of control code. This varies help function to create correct size of cache tensor
        src (int, optional): Source rank. Will receive from any process if unspecified.

    Returns:
        (rank of sender, list of control codes, serialized model parameters)
    """
    print("RECV MESSAGE: RANK: {}".format(dist.get_rank()))  # debug

    # create a tensor to save recieved information
    cache_tensor = torch.zeros(size=(control_code_size+sp_size,))
    dist.recv(tensor=cache_tensor, src=src)

    # this function dose not manipulate model prameters!
    # return parsed information: sender rank, control codes, serialized model parameters
    return cache_tensor[0], cache_tensor[1:control_code_size+1], cache_tensor[control_code_size+1:]


def send_message(message_code, payload, dst=0):
    """Sends a message to destination.

    Concatenates destination rank, message code and payload into a single tensor, then sends it.

    Args:
        message_code (MessageCode): Type of message, defined in MessageCode
        payload: Serialized tensor
        dst (int, optional): Destination rank. Default is 0 which means the server

    Returns:
        Serialized message package, including rank of current process rank, message code, and the payload
    """
    # _LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
    print("SENDING MESSAGE: {} RANK: {} => RANK: {}".format(
        message_code, dist.get_rank(), dst))
    # tensor shape is torch.Size([2])
    m_parameter = torch.Tensor([dist.get_rank(), message_code.value])
    # use cpu to improve IO efficiency
    m_parameter = torch.cat((m_parameter, payload.cpu()))
    # 使用isend，线程退出或者其他原因，将导致发送失败
    dist.send(tensor=m_parameter, dst=dst)
    return m_parameter


def recv_message(payload, src=None):
    """Receives message from source.

    Args:
        payload (torch.Tensor): Tensor to fill with received data. The first element is the source rank, and the second element is
    message code.
        src (int, optional): Source rank. Will receive from any process if unspecified.

    Returns:
        Serialized model parameters
    """
    print("RECV MESSAGE: RANK: {}".format(dist.get_rank()))  # debug
    dist.recv(tensor=payload, src=src)
    return payload[2:]
