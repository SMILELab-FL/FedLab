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

class Compressor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compress(self, *args, **kwargs):
        pass

    @abstractmethod
    def decompress(self, *args, **kwargs):
        pass

class Memory(ABC):
    @staticmethod
    def initialize(*args, **kwargs):
        pass

    @staticmethod
    def compensate(tensor, *args, **kwargs):
        return tensor

    @staticmethod
    def update(*args, **kwargs):
        pass

    @staticmethod
    def state_dict():
        return None

    @staticmethod
    def load_state_dict(state_dict):
        pass
