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

import math
import torch

from .compressor import Compressor


class TopkCompressor(Compressor):
    """ Compressor for federated communication
        Top-k gradient or weights selection
        Args:
            compress_ratio (float): compress ratio
    """
    def __init__(self, compress_ratio):
        self.compress_ratio = compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.index_dtype = torch.int64
        self.value_dtype = torch.float32

    def compress(self, tensor):
        """compress tensor into (values, indices)
        Args:
            tensor (torch.Tensor): tensor
        Returns:
            tuple: (values, indices)
        """
        if torch.is_tensor(tensor):
            tensor = tensor.detach()
        else:
            raise TypeError(
                "Invalid type error, expecting {}, but get {}".format(
                    torch.Tensor, type(tensor)))

        numel = tensor.numel()
        top_k_samples = int(math.ceil(numel * self.compress_ratio))

        tensor = tensor.view(-1)
        importance = tensor.abs()

        _, indices = torch.topk(importance,
                                top_k_samples,
                                0,
                                largest=True,
                                sorted=False)
        values = tensor[indices]

        values = values.to(dtype=self.value_dtype)
        indices = indices.to(dtype=self.index_dtype)

        return values, indices

    def decompress(self, values, indices, shape):
        """decompress tensor"""
        de_tensor = torch.zeros(size=shape, dtype=self.value_dtype).view(-1)
        de_tensor = de_tensor.index_put_([indices], values,
                                         accumulate=True).view(shape)
        return de_tensor
        