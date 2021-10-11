# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import e
import unittest
import torch
import torch.distributed as dist
import os

from random import randint
from fedlab.core.communicator.package import Package
from fedlab.utils.message_code import MessageCode
from fedlab.core.communicator.package import (
    HEADER_SENDER_RANK_IDX, HEADER_RECEIVER_RANK_IDX, HEADER_SLICE_SIZE_IDX,
    HEADER_MESSAGE_CODE_IDX, DEFAULT_SLICE_SIZE,
    DEFAULT_MESSAGE_CODE_VALUE, HEADER_SIZE, HEADER_DATA_TYPE_IDX)


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

        cls.content = torch.cat(cls.tensor_list)
        cls.slice_size = cls.tensor_num

    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        # initialize the process group
        # GitHub Actions only support CPU, so only backend "gloo" for CPU can be used for test initialization
        world_size = 1
        dist.init_process_group(backend="gloo",
                                rank=self.sender_rank,
                                world_size=world_size)

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _assert_tensor_eq(self, t1, t2):
        self.assertTrue(torch.equal(t1, t2))

    def test_pack_up_default(self):
        p = Package()

        assert p.content == None
        assert p.header[HEADER_SENDER_RANK_IDX] == dist.get_rank()
        assert p.header[HEADER_RECEIVER_RANK_IDX] == -1
        assert p.header[HEADER_MESSAGE_CODE_IDX] == DEFAULT_MESSAGE_CODE_VALUE
        assert p.header[HEADER_SLICE_SIZE_IDX] == DEFAULT_SLICE_SIZE

    def test_pack_up_with_content(self):
        # init with single tensor content
        p1 = Package(content=self.tensor_list[0])

        self._assert_tensor_eq(p1.content, self.tensor_list[0])

        # init with tensor list content
        p2 = Package(content=self.tensor_list)
        self._assert_tensor_eq(p2.content, self.content)
        self.assertEqual(int(p2.header[HEADER_SLICE_SIZE_IDX]),
                         self.slice_size)


    def test_add_tensor(self):
        p = Package()
        p.append_tensor(self.tensor_list[0])

        assert p.header[HEADER_SLICE_SIZE_IDX] == len(p.slices)
        assert len(p.slices) == 1
        assert sum(p.slices) == p.content.shape[0]

    def test_add_tensor_list(self):
        p = Package()
        p.append_tensor_list(self.tensor_list)

        assert p.header[HEADER_SLICE_SIZE_IDX] == len(p.slices)
        assert len(p.slices) == len(self.tensor_list)
        assert sum(p.slices) == p.content.shape[0]

    def test_parse_content(self):
        p = Package()
        p.append_tensor_list(self.tensor_list)

        slices = p.slices

        parsed_content = Package.parse_content(slices, p.content)

        for t, p_t in zip(self.tensor_list, parsed_content):
            self._assert_tensor_eq(t, p_t)
        
    def test_parse_header(self):
        p = Package()
        p.append_tensor_list(self.tensor_list)

        sender_rank, receiver_rank, slice_size, message_code, data_type = Package.parse_header(p.header)

        assert sender_rank == p.header[HEADER_SENDER_RANK_IDX]
        assert receiver_rank == p.header[HEADER_RECEIVER_RANK_IDX]
        assert slice_size == p.header[HEADER_SLICE_SIZE_IDX]
        assert message_code.value == p.header[HEADER_MESSAGE_CODE_IDX]
        assert data_type == p.header[HEADER_DATA_TYPE_IDX]