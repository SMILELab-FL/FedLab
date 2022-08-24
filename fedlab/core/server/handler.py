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

from abc import abstractproperty
from typing import List

import torch

from ..model_maintainer import ModelMaintainer


class ServerHandler(ModelMaintainer):
    """An abstract class representing handler of parameter server.

    Please make sure that your self-defined server handler class subclasses this class

    Example:
        Read source code of :class:`SyncServerHandler` and :class:`AsyncServerHandler`.
    """

    def __init__(self, model, cuda=False):
        super().__init__(model, cuda)

    @abstractproperty
    def downlink_package(self) -> List[torch.Tensor]:
        """Property for manager layer. Server manager will call this property when activates clients."""
        raise NotImplementedError()

    @abstractproperty
    def if_stop(self) -> bool:
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return False

    def global_update(self, buffer):
        raise NotImplementedError()

    def load(self, payload):
        """Override this function to define how to update global model (aggregation or optimization)."""
        raise NotImplementedError()

