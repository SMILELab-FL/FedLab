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
from fedlab.core.communicator import dtype_torch2flab, dtype_flab2torch
from fedlab.core.communicator import (
    INT8, INT16, INT32, INT64, FLOAT16, FLOAT32, FLOAT64
)
from fedlab.core.communicator.package import Package, supported_torch_dtypes
from fedlab.utils.message_code import MessageCode
from fedlab.core.communicator.package import (
    HEADER_SENDER_RANK_IDX, HEADER_RECEIVER_RANK_IDX, HEADER_SLICE_SIZE_IDX,
    HEADER_MESSAGE_CODE_IDX, DEFAULT_SLICE_SIZE, DEFAULT_MESSAGE_CODE_VALUE,
    HEADER_SIZE, HEADER_DATA_TYPE_IDX)

class DtypeFuncTestCase(unittest.TestCase):
    def setUp(self):
        self.fedlab_types = [INT8, INT16, INT32, INT64, FLOAT16, FLOAT32, FLOAT64] 

    def test_dtype_flab2torch(self):
        for fedlab_dtype in self.fedlab_types:
            res = dtype_flab2torch(fedlab_dtype)
    
    def test_dtype_torch2flab(self):
        for torch_dtype in supported_torch_dtypes:
            res = dtype_torch2flab(torch_dtype)

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
        os.environ["MASTER_PORT"] = "12356"
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
        self.assertEqual(p.content, None)
        self.assertEqual(p.header[HEADER_SENDER_RANK_IDX], dist.get_rank())
        self.assertEqual(p.header[HEADER_RECEIVER_RANK_IDX], -1)
        self.assertEqual(p.header[HEADER_MESSAGE_CODE_IDX], DEFAULT_MESSAGE_CODE_VALUE)
        self.assertEqual(p.header[HEADER_SLICE_SIZE_IDX], DEFAULT_SLICE_SIZE)

    def test_pack_up_with_content(self):
        # init with single tensor content
        p1 = Package(content=self.tensor_list[0])

        self._assert_tensor_eq(p1.content, self.tensor_list[0])

        # init with tensor list content
        p2 = Package(content=self.tensor_list)
        self._assert_tensor_eq(p2.content, self.content)

    def test_pack_up_with_MessageCode(self):
        # init with message_code as MessageCode
        for msg_code in MessageCode:
            p = Package(message_code=msg_code)
            self.assertEqual(p.header[HEADER_MESSAGE_CODE_IDX], msg_code.value)

    def test_append_tensor(self):
        p = Package()
        p.append_tensor(self.tensor_list[0])
        self.assertEqual(p.header[HEADER_SLICE_SIZE_IDX], len(p.slices))

    def test_append_tensor_diff_dtype(self):
        p = Package()
        p.append_tensor(self.tensor_list[0])  # set p.dtype=torch.fload32 by setting first tensor
        with self.assertWarns(UserWarning):
            int_tensor = torch.tensor([1,2], dtype=int)
            p.append_tensor(int_tensor)  # append torch.int64 tensor


    def test_append_tensor_invalid(self):
        p = Package()
        with self.assertRaises(ValueError):
            p.append_tensor({'a':1, 'b': 2, 'c': 3})  # use dict as input tensor, rather than torch.Tensor

    def test_append_tensor_list(self):
        p = Package()
        p.append_tensor_list(self.tensor_list)
        self.assertEqual(p.header[HEADER_SLICE_SIZE_IDX], len(p.slices))

    def test_to_supported_dtype(self):
        p = Package(content=self.tensor_list[0])
        new_dtype = torch.float64
        self.assertNotEqual(p.dtype, new_dtype)  # p.dtype=torch.float32 after init
        p.to(new_dtype)  # set to torch.float64
        self.assertEqual(p.dtype, new_dtype)
        self.assertEqual(p.content.dtype, new_dtype)

    def test_to_unsupported_dtype(self):
        p = Package(content=self.tensor_list[0])
        new_dtype = type({'a': 1, 'b': 2})  # dict type
        self.assertNotEqual(p.dtype, new_dtype)  # p.dtype=torch.float32 after init
        with self.assertWarns(UserWarning):
            p.to(new_dtype)  # set to dict
            self.assertNotEqual(p.dtype, new_dtype)

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

        sender_rank, receiver_rank, slice_size, message_code, data_type = Package.parse_header(
            p.header)

        self.assertEqual(sender_rank, p.header[HEADER_SENDER_RANK_IDX])
        self.assertEqual(receiver_rank, p.header[HEADER_RECEIVER_RANK_IDX])
        self.assertEqual(slice_size, p.header[HEADER_SLICE_SIZE_IDX])
        self.assertEqual(message_code.value, p.header[HEADER_MESSAGE_CODE_IDX])
        self.assertEqual(data_type, p.header[HEADER_DATA_TYPE_IDX])