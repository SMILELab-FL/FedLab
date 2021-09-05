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

from abc import ABC, abstractmethod
import math
import torch
from .serialization import SerializationTool

class Compressor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compress_tensor(self, *args, **kwargs):
        pass

    @abstractmethod
    def decompress_tensor(self, *args, **kwargs):
        pass

    @abstractmethod
    def compress_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def decompress_model(self, *args, **kwargs):
        pass


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
            tuple: values, indices
        """
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

    def compress_model(self, model):
        """compress model

        Args:
            model (nn.module): PyTorch module.

        Returns:
            tuple: list(values) and list(indices).
        """
        model_values = []
        model_indices = []
        for parameter in model.parameters():
            values, indices = self.compress_tensor(parameter)
            model_values.append(values)
            model_indices.append(indices)

        return model_values, model_indices

    def decompress_model(self, model, model_values, model_indices):
        """decompress model

        Args:
            model (nn.module): PyTorch module.
            model_values (list[torch.Tensor]): values.
            model_indices (list[torch.Tensor]): indices.
        """
        model_parameters_layer_list = []
        for parameter, values, indices in zip(model.parameters(), model_values,
                                              model_indices):
            de_tensor = self.decompress_tensor(values, indices,
                                               parameter.shape)
            model_parameters_layer_list.append(de_tensor.view(-1))
        
        model_parameters = torch.cat(model_parameters_layer_list)
        SerializationTool.deserialize_model(model,model_parameters)
