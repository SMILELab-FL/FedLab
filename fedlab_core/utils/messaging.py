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
    """Sends a message to a destination
    Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor

    Args:
        message_code: defined in MessageCode
        payload: serialized tensor
        dst: destination this message go, default is 0 which means server

    Returns:
        serialized message package

    Raises:
        None
    """
    #_LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
    print("SENDING MESSAGE: {} RANK: {} => RANK: {}".format(
        message_code, dist.get_rank(), dst))
    m_parameter = torch.Tensor([dist.get_rank(), message_code.value])
    m_parameter = torch.cat((m_parameter, payload.cpu()))
    # 使用isend，线程退出或者其他原因，将导致发送失败
    dist.send(tensor=m_parameter, dst=dst)
    return m_parameter


def recv_message(payload, src=None):
    """Receive meassage from src
        when src is None, process will receive from whoever send message

    Args:
        payload: given tensor to store the data this process received
        src: sender index
        
    Returns:
        serialized model parameters

    Raises:
        None
    """
    #_LOGGER.info("RECV MESSAGE: RANK: {}".format(dist.get_rank()))
    print("RECV MESSAGE: RANK: {}".format(dist.get_rank()))
    dist.recv(tensor=payload, src=src)
    return payload[2:]


def broadcast_message(message_code, payload):
    """broadcast a message to all workers
    Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor

    Args:

    Returns:

    Raises:
        
    """
    #_LOGGER.info("SBROADCASTING MESSAGE: {} RANK: {} => ALL".format(message_code, dist.get_rank()))
    # print("BROADCASTING MESSAGE: {} RANK: {} => ALL ".format(message_code, dist.get_rank()))
    m_parameter = torch.Tensor([dist.get_rank(), message_code.value])
    m_parameter = torch.cat((m_parameter, payload))
    # 使用isend，线程退出或者其他原因，将导致发送失败
    dist.broadcast(tensor=m_parameter, src=0)


def recv_broadcast_message(recv_buff):
    """workers recv the message from the center
    
    Args:

    Returns:

    Raises:
        
    """
    dist.broadcast(tensor=recv_buff, src=0)
    #message_code = MessageCode(recv_buff[1].item()),
    #_LOGGER.info("RECVING BROADCAST MESSAGE: {} FROM RANK: {}".format(message_code, dist.get_rank()))
