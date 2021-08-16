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


class Compressor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compress_tensor(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def decompress_tensor(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def compress_model(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def decompress_model(self, *args, **kwargs):
        raise NotImplementedError()


class TopkCompressor(Compressor):
    """ Compressor for federated communication

        Top-k gradient or weights selection

        Args:
            compress_ratio (float): compress ratio
    """
    def __init__(self, compress_ratio):
        self.compress_ratio = compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio

    def compress_tensor(self, tensor):
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
        de_tensor = torch.zeros(size=shape).view(-1)
        de_tensor = de_tensor.index_put_([indices],
                                            values,
                                            accumulate=True).view(shape)
        return de_tensor

    def compress_model(self, model):
        model_values = []
        model_indices = []
        for parameter in model.parameters():
            values, indices = self.compress_tensor(parameter)
            model_values.append(values)
            model_indices.append(indices)

        #model_values = torch.cat(model_values)
        #model_indices = torch.cat(model_indices)
        return model_values, model_indices

    def decompress_model(self, model_values, model_indices):
        
        pass
