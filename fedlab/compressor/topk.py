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

from . import Compressor
from ..utils.serialization import SerializationTool

class TopkCompressor(Compressor):
    """ Compressor for federated communication

        Top-k gradient or weights selection

        Args:
            compress_ratio (float): compress ratio
    """
    def __init__(self, compress_ratio):
        self.compress_ratio = compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio

    def compress_tensor(self, tensor):
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

        return values, indices

    def decompress_tensor(self, values, indices, shape):
        """decompress tensor"""
        de_tensor = torch.zeros(size=shape).view(-1)
        de_tensor = de_tensor.index_put_([indices], values,
                                         accumulate=True).view(shape)
        return de_tensor

    def compress(self, parameters):
        """compress model

        Args:
            model (nn.module): PyTorch module.

        Returns:
            tuple: list(values) and list(indices).
        """
        values_list = []
        indices_list = []
        for param in parameters:
            values, indices = self.compress_tensor(param)
            values_list.append(values)
            indices_list.append(indices)

        return values_list, indices_list

    def decompress(self, shape_list, values_list, indices_list):
        """decompress model

        Args:
            shape_list (list[tuple]): The shape of every corresponding tensor.
            values_list (list[torch.Tensor]): list(values).
            indices_list (list[torch.Tensor]): list(indices).
        """
        parameters_layer_list = []
        for shape, values, indices in zip(shape_list, values_list, indices_list):
            de_tensor = self.decompress_tensor(values, indices, shape)
            parameters_layer_list.append(de_tensor.view(-1))

        parameters = torch.cat(parameters_layer_list)

        return parameters
