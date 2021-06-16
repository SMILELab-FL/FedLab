# 参考dgc中DGCCompressor的实现形式
from abc import ABC, abstractmethod
import math
import random
import torch
from torch._C import DoubleTensor


class Compressor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initialize(self, named_parameters):
        raise NotImplementedError()

    @abstractmethod
    def compress(self):
        raise NotImplementedError()

    @abstractmethod
    def decompress(self):
        raise NotImplementedError()


class TopkCompressor(Compressor):
    def __init__(self, compress_ratio, fp16_values=False, int32_indices=False):

        self.fp16_values = fp16_values
        self.int32_indices = int32_indices

        self.compress_ratio = compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio

        self.attributes = {}  # 定义压缩矩阵的基本信息： [numel, shape, num_selects, num_samples, top_k_samples, sample_stride]

    def initialize(self, named_parameters):
        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]

            top_k_samples = int(math.ceil(numel * self.compress_ratio))
            self.attributes[name] = (numel, shape, top_k_samples)

    def compress(self, tensor, name):
        # 对于已注册的数据结构/模型参数压缩
        if self.compress_ratio < 1.0 and name in self.attributes:
            tensor = tensor.view(-1)
        
            numel, shape, top_k_samples = self.attributes[name]

            importance = tensor.abs()
            _, indices = torch.topk(importance,
                                    top_k_samples,
                                    0,
                                    largest=True,
                                    sorted=False)
            values = tensor[indices]
            return (values, indices), (name, shape)
        else:
            raise ValueError("invalid value")

    def decompress(self, tensor, ctx):
        name, shape = ctx
        if self.compress_ratio < 1.0 and name in self.attributes:
            values, indices = tensor

            de_tensor = torch.zeros(size=shape).index_put_([indices],
                                                           values,
                                                           accumulate=True)
            return de_tensor
        else:
            raise ValueError("invalid value")