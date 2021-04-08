import logging
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
    m_parameter = torch.Tensor([dist.get_rank(), message_code.value])  # tensor shape is torch.Size([2])
    m_parameter = torch.cat((m_parameter, payload.cpu()))  # TODO: why force to use cpu()?
    # 使用isend，线程退出或者其他原因，将导致发送失败
    dist.send(tensor=m_parameter, dst=dst)
    return m_parameter


def recv_message(payload, src=None):
    """Receives message from source.

    Args:
        payload: Tensor to fill with received data. The first element is the source rank, and the second element is
        message code.
        src (int, optional): Source rank. Will receive from any process if unspecified.
        
    Returns:
        Serialized model parameters

    Raises:
        None
    """
    # _LOGGER.info("RECV MESSAGE: RANK: {}".format(dist.get_rank()))
    print("RECV MESSAGE: RANK: {}".format(dist.get_rank()))
    dist.recv(tensor=payload, src=src)
    return payload[2:]


def broadcast_message(message_code, payload):
    """Broadcast a message to all workers.

    Concatenates destination rank, message code and payload into a single tensor, then broadcasts the tensor to the
    whole group.

    Args:

    Returns:

    Raises:
        
    """
    # _LOGGER.info("SBROADCASTING MESSAGE: {} RANK: {} => ALL".format(message_code, dist.get_rank()))
    # print("BROADCASTING MESSAGE: {} RANK: {} => ALL ".format(message_code, dist.get_rank()))
    m_parameter = torch.Tensor([dist.get_rank(), message_code.value])
    m_parameter = torch.cat((m_parameter, payload))
    # 使用isend，线程退出或者其他原因，将导致发送失败
    dist.broadcast(tensor=m_parameter, src=0)


def recv_broadcast_message(recv_buff):
    """Workers recv the message from the center
    
    Args:

    Returns:

    Raises:
        
    """
    dist.broadcast(tensor=recv_buff, src=0)
    # message_code = MessageCode(recv_buff[1].item()),
    # _LOGGER.info("RECVING BROADCAST MESSAGE: {} FROM RANK: {}".format(message_code, dist.get_rank()))
