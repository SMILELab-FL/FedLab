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

from abc import abstractmethod
from typing import List

import torch

from ..model_maintainer import ModelMaintainer


class ServerHandler(ModelMaintainer):
    """An abstract class representing handler of parameter server.

    Please make sure that your self-defined server handler class subclasses this class

    Example:
        Read source code of :class:`SyncServerHandler` and :class:`AsyncServerHandler`.
        
    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool): Use GPUs or not.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest memory as default.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 cuda: bool,
                 device: str = None) -> None:
        super().__init__(model, cuda, device)

    @property
    @abstractmethod
    def downlink_package(self) -> List[torch.Tensor]:
        """Property for manager layer. Server manager will call this property when activates clients."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def if_stop(self) -> bool:
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return False

    @abstractmethod
    def setup_optim(self):
        """Override this function to load your optimization hyperparameters."""
        raise NotImplementedError()

    @abstractmethod
    def global_update(self, buffer):
        raise NotImplementedError()

    @abstractmethod
    def load(self, payload):
        """Override this function to define how to update global model (aggregation or optimization)."""
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self):
        """Override this function to define the evaluation of global model."""
        raise NotImplementedError()
