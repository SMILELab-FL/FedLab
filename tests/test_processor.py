# -*- coding: utf-8 -*-
# @Time    : 4/27/21 8:27 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : test_processor.py
# @Software: PyCharm
import unittest
import os
from random import randint

from fedlab_core.communicator.package import (HEADER_SENDER_RANK_IDX,
                                                HEADER_RECEIVER_RANK_IDX,
                                                HEADER_CONTENT_SIZE_IDX,
                                                HEADER_MESSAGE_CODE_IDX,
                                                DEFAULT_RECEIVER_RANK,
                                                DEFAULT_CONTENT_SIZE,
                                                DEFAULT_MESSAGE_CODE_VALUE,
                                                HEADER_SIZE)

from fedlab_core.communicator.package import Package
from fedlab_utils.message_code import MessageCode

import torch
import torch.distributed as dist


class PackageTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sender_rank = 0
        cls.recver_rank = 1
        cls.message_code = MessageCode.EvaluateParams
        cls.message_code_value = cls.message_code.value

        cls.tensor_num = 5
        cls.tensor_sizes = [randint(3, 15) for _ in range(cls.tensor_num)]
        cls.tensor_list = [torch.rand(size) for size in cls.tensor_sizes]
        tmp_content = [torch.cat((torch.Tensor([tensor_size]), tensor)) for tensor, tensor_size in
                       zip(cls.tensor_list, cls.tensor_sizes)]
        cls.content = torch.cat(tmp_content)
        cls.content_size = sum(cls.tensor_sizes) + cls.tensor_num

    def setUp(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        # initialize the process group
        # GitHub Actions only support CPU, so only backend "gloo" for CPU can be used for test initialization
        world_size = 1
        dist.init_process_group(backend="gloo", rank=self.sender_rank, world_size=world_size)

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _assert_tensor_eq(self, t1, t2):
        self.assertTrue(torch.equal(t1, t2))

    def test_init_package_default(self):
        p = Package()
        self._assert_tensor_eq(p.header, torch.Tensor([dist.get_rank(),
                                                       DEFAULT_RECEIVER_RANK,
                                                       DEFAULT_CONTENT_SIZE,
                                                       DEFAULT_MESSAGE_CODE_VALUE]))
        self._assert_tensor_eq(p.content, torch.zeros(size=(1,)))
        self.assertEqual(p.content_flag, False)

    def test_init_package_with_content(self):
        # init with single tensor content
        p1 = Package(content=self.tensor_list[0])
        self._assert_tensor_eq(p1.content, torch.cat([torch.Tensor([self.tensor_sizes[0]]), self.tensor_list[0]]))
        self.assertEqual(p1.content_flag, True)
        self.assertEqual(int(p1.header[HEADER_CONTENT_SIZE_IDX]), self.tensor_sizes[0] + 1)

        # init with tensor list content
        p2 = Package(content=self.tensor_list)
        self._assert_tensor_eq(p2.content, self.content)
        self.assertEqual(int(p2.header[HEADER_CONTENT_SIZE_IDX]), self.content_size)

    def test_init_package_without_content(self):
        p1 = Package(receiver_rank=self.recver_rank, message_code=self.message_code)
        h1 = torch.Tensor(size=(HEADER_SIZE,))
        h1[HEADER_RECEIVER_RANK_IDX] = self.recver_rank
        h1[HEADER_SENDER_RANK_IDX] = dist.get_rank()
        h1[HEADER_CONTENT_SIZE_IDX] = DEFAULT_CONTENT_SIZE
        h1[HEADER_MESSAGE_CODE_IDX] = self.message_code.value
        self._assert_tensor_eq(p1.header, h1)

        p2 = Package(receiver_rank=self.recver_rank, message_code=self.message_code_value)
        h2 = torch.Tensor(size=(HEADER_SIZE,))
        h2[HEADER_RECEIVER_RANK_IDX] = self.recver_rank
        h2[HEADER_SENDER_RANK_IDX] = dist.get_rank()
        h2[HEADER_CONTENT_SIZE_IDX] = DEFAULT_CONTENT_SIZE
        h2[HEADER_MESSAGE_CODE_IDX] = self.message_code_value
        self._assert_tensor_eq(p2.header, h2)

        with self.assertRaises(AssertionError):
            p3 = Package(message_code=3.5)
        with self.assertRaises(AssertionError):
            p4 = Package(receiver_rank=4.5)

    def test_append_tensor(self):
        p = Package()
        self.assertEqual(p.content_flag, False)
        p.append_tensor(self.tensor_list[0])
        self._assert_tensor_eq(p.content, torch.cat([torch.Tensor([self.tensor_sizes[0]]), self.tensor_list[0]]))
        self.assertEqual(int(p.header[HEADER_CONTENT_SIZE_IDX]), self.tensor_sizes[0] + 1)
        self.assertEqual(p.content_flag, True)

    def test_append_tensor_list(self):
        p = Package()
        self.assertEqual(p.content_flag, False)
        p.append_tensor_list(self.tensor_list)
        self._assert_tensor_eq(p.content, self.content)
        self.assertEqual(p.header[HEADER_CONTENT_SIZE_IDX], self.content_size)
        self.assertEqual(p.content_flag, True)

    def test_parse_content_with_content(self):
        p = Package(content=self.tensor_list)
        content_parse_res = Package.parse_content(p.content)
        self.assertEqual(len(content_parse_res), self.tensor_num)
        for t_res, t in zip(content_parse_res, self.tensor_list):
            self._assert_tensor_eq(t_res, t)

    def test_parse_content_without_content(self):
        p = Package()
        content_parse_res = Package.parse_content(p.content)
        self.assertEqual(content_parse_res, [])

    def test_parse_header(self):
        # test on package with empty content
        p_no_content = Package(receiver_rank=self.recver_rank, message_code=self.message_code_value)
        sender_rank, recver_rank, content_size, message_code = Package.parse_header(p_no_content.header)
        self.assertEqual(sender_rank, dist.get_rank())
        self.assertEqual(recver_rank, self.recver_rank)
        self.assertEqual(content_size, DEFAULT_CONTENT_SIZE)
        self.assertEqual(message_code, self.message_code)

        # test on package with empty content
        p_with_content = Package(receiver_rank=self.recver_rank, message_code=self.message_code_value,
                                 content=self.tensor_list)
        sender_rank, recver_rank, content_size, message_code = Package.parse_header(p_with_content.header)
        self.assertEqual(sender_rank, dist.get_rank())
        self.assertEqual(recver_rank, self.recver_rank)
        self.assertEqual(content_size, sum(self.tensor_sizes) + self.tensor_num)
        self.assertEqual(message_code, self.message_code)
