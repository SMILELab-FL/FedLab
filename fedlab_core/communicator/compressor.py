#TODO: 开放的模型压缩接口
# 参考dgc中DGCCompressor的实现形式

from abc import ABC, abstractmethod


class Compressor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compress(self):
        pass
    
    
    @abstractmethod
    def decompress(self):
        pass
