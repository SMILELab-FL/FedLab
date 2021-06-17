import torch
import torch.distributed as dist
from fedlab_utils.serialization import SerializationTool
from fedlab_utils.message_code import MessageCode
from fedlab_core.communicator import package
from fedlab_core.communicator.package import Package


class PackageProcessor(object):
    """Provide more flexible distributed tensor communication functions based on :func:`torch.distributed.send` and
    :func:`torch.distributed.recv`"""
    @staticmethod
    def recv_package(src=None):
        def recv_header(src=src, parse=True):
            buffer = torch.zeros(size=(package.HEADER_SIZE, ))
            dist.recv(buffer, src=src)
            if parse is True:
                return Package.parse_header(buffer)
            else:
                return buffer

        def recv_content(cache_size, src):
            buffer = torch.zeros(size=(cache_size, ))
            dist.recv(buffer, src=src)
            return Package.parse_content(buffer)

        sender_rank, recv_rank, content_size, message_code = recv_header(
            src=src)

        # 收到第一段包，第二段包指定来源rank
        if content_size > 0:
            content = recv_content(content_size, src=sender_rank)
        else:
            content = None

        return sender_rank, message_code, content

    @staticmethod
    def send_package(package, dst):
        """Two-segment tensor communication pattern based on ``torch.distributed``

        Pattern is shown as follows:
            1.1 sender: send a header tensor containing ``content_size`` to receiver
            1.2 receiver: receive the header, and get the value of ``content_size`` and create a buffer for incoming content

            2.1 sender: send a content tensor composed of a list of tensors and their offsets
            2.2 receiver: receive the content tensor, and parse it to obtain a tensor list using parser function
        """
        def send_header(header, dst):
            header[package.HEADER_RECEIVER_RANK_IDX] = dst
            dist.send(header, dst=dst)

        def send_content(content, dst):
            dist.send(content, dst=dst)

        send_header(header=package.header, dst=dst)

        if package.header[package.HEADER_CONTENT_SIZE_IDX] > 0:
            send_content(content=package.content, dst=dst)
