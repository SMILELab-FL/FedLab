# -*- coding: utf-8 -*-
# @Time    : 4/27/21 8:27 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : test_processor.py
# @Software: PyCharm
import unittest
import os
from random import randint

from fedlab_core.communicator.processor import (HEADER_SENDER_RANK_IDX,
                                                HEADER_RECVER_RANK_IDX,
                                                HEADER_CONTENT_SIZE_IDX,
                                                HEADER_MESSAGE_CODE_IDX,
                                                DEFAULT_RECVER_RANK,
                                                DEFAULT_CONTENT_SIZE,
                                                DEFAULT_MC,
                                                HEADER_SIZE)

from fedlab_core.communicator.processor import Package
from fedlab_core.communicator.processor import PackageProcessor
from fedlab_utils.message_code import MessageCode

import torch
import torch.distributed as dist


class PackageTestCase(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls) -> None:
    #     cls.default_header = torch.Tensor()
    # @classmethod
    # def setUpClass(cls) -> None:
    #     message_code = MessageCode.ParameterUpdate
    #     cls.none_header_full_content_package = Package(message_code, header=None, content=None)
    #     cls.full_header_none_content_package = Package(message_code, header=None, content=None)
    #     cls.none_header_none_content_package = Package(message_code, header=None, content=None)
    #     cls.full_header_full_content_package = Package(message_code, header=None, content=None)

    def setUp(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        rank = 0
        world_size = 1
        # initialize the process group
        # GitHub Actions only support CPU, so only backend "gloo" for CPU can be used for initialization
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        message_code = MessageCode.ParameterUpdate

        self.list_header = [rank, 1, 10, message_code]
        self.default_header = torch.Tensor([rank, DEFAULT_RECVER_RANK, DEFAULT_CONTENT_SIZE, message_code])
        self.default_content = torch.zeros(size=(1,))

        self.none_header_full_content_package = Package(message_code, header=None, content=None)
        self.full_header_none_content_package = Package(message_code, header=self.list_header, content=None)
        self.none_header_none_content_package = Package(message_code, header=None, content=None)
        self.full_header_full_content_package = Package(message_code, header=self.list_header, content=None)

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_append_tensor(self):
        tensor_size = 10
        tensor = torch.rand(tensor_size)
        pass

    def test_append_tensor_list(self):
        tensor_num = 5
        tensor_sizes = [randint(3, 15) for _ in range(tensor_num)]
        tensor_list = [torch.rand(size) for size in tensor_sizes]
        tmp_content = [torch.cat((tensor, tensor_size)) for tensor, tensor_size in zip(tensor_list, tensor_sizes)]
        content = torch.cat(tmp_content)
        content_size = sum(tensor_sizes) + tensor_num
        pass
