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

from abc import abstractclassmethod, abstractproperty
from typing import List

import torch

from fedlab.dataset.dataset import FedLabDataset

from ..client import ORDINARY_TRAINER, SERIAL_TRAINER
from ..model_maintainer import ModelMaintainer, SerialModelMaintainer
from ...utils import Logger, SerializationTool


class ClientTrainer(ModelMaintainer):
    """An abstract class representing a client trainer.

    In FedLab, we define the backend of client trainer show manage its local model.
    It should have a function to update its model called :meth:`local_process`.

    If you use our framework to define the activities of client, please make sure that your self-defined class
    should subclass it. All subclasses should overwrite :meth:`local_process` and property `uplink_package`.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool): Use GPUs or not.
    """

    def __init__(self, model, cuda, device=None) -> None:
        super().__init__(model, cuda, device)

        self.client_num = 1  # default is 1.
        self.dataset = self.__setup_dataset()
        self.type = ORDINARY_TRAINER

    def __setup_dataset(self):
        """Override this function to set up local dataset for clients"""
        return FedLabDataset()

    @abstractproperty
    def uplink_package(self) -> List[torch.Tensor]:
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        raise NotImplementedError()

    @abstractclassmethod
    def local_process(self, payload) -> bool:
        """Manager of the upper layer will call this function with accepted payload
        
            In synchronous mode, return True to end current FL round.
        """
        raise NotImplementedError()

    def train(self):
        """Override this method to define the algorithm of training your model. This function should manipulate :attr:`self._model`"""
        raise NotImplementedError()

    def evaluate(self):
        """Evaluate quality of local model."""
        raise NotImplementedError()

class SerialClientTrainer(SerialModelMaintainer):
    """Base class. Simulate multiple clients in sequence in a single process.

    Args:
        model (torch.nn.Module): Model used in this federation.
        client_num (int): Number of clients in current trainer.
        cuda (bool): Use GPUs or not. Default: ``False``.
    """

    def __init__(self, model, num, cuda, device=None) -> None:
        super().__init__(model, num, cuda, device)

        self.client_num = 1  # default is 1.
        self.dataset = self.__setup_dataset()
        self.type = SERIAL_TRAINER  # represent serial trainer

    def __setup_dataset(self):
        """Override this function to set up local dataset for clients"""
        return FedLabDataset()

    @abstractproperty
    def uplink_package(self) -> List[torch.Tensor]:
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        raise NotImplementedError()


    @abstractclassmethod
    def local_process(self, id_list, payload) -> bool:
        """Manager of the upper layer will call this function with accepted payload
        
            In synchronous mode, return True to end current FL round.
        """
        raise NotImplementedError()

    def train(self):
        """Override this method to define the algorithm of training your model. This function should manipulate :attr:`self._model`"""
        raise NotImplementedError()

    def evaluate(self):
        """Evaluate quality of local model."""
        raise NotImplementedError()
