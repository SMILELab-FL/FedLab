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

import unittest

from fedlab_utils.models.lenet import LeNet
from fedlab_core.communicator.compressor import TopkCompressor


class CompressorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LeNet()
        self.compressor = TopkCompressor(compress_ratio=0.5)
        self.compressor.initialize(self.model.named_parameters())

        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_compress(self):
        for name, param in self.model.named_parameters():
            tensor_info, ctx = self.compressor.compress(param, name)
            decompressed = self.compressor.decompress(tensor_info, ctx)

            assert decompressed.shape == param.shape
