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

from abc import abstractclassmethod, abstractproperty, abstractmethod
from random import randint
from typing import List

import torch

from fedlab.contrib.dataset.basic_dataset import FedDataset

from ..client import ORDINARY_TRAINER, SERIAL_TRAINER
from ..model_maintainer import ModelMaintainer, SerialModelMaintainer
from ...utils import Logger, SerializationTool


class ClientTrainer(ModelMaintainer):
    """An abstract class representing a client trainer.

    In FedLab, we define the backend of client trainer show manage its local model.
    It should have a function to update its model called :meth:`local_process`.

    If you use our framework to define the activities of client, please make sure that your self-defined class
    should subclass it. All subclasses should overwrite :meth:`local_process` and property ``uplink_package``.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool): Use GPUs or not.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to ``None``.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 cuda: bool,
                 device: str = None) -> None:
        super().__init__(model, cuda, device)

        self.num_clients = 1  # default is 1.
        self.dataset = FedDataset() # or Dataset
        self.type = ORDINARY_TRAINER

    def setup_dataset(self):
        """Set up local dataset ``self.dataset`` for clients."""
        raise NotImplementedError()

    def setup_optim(self):
        """Set up variables for optimization algorithms."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def uplink_package(self) -> List[torch.Tensor]:
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        raise NotImplementedError()

    @abstractclassmethod
    def local_process(self, payload: List[torch.Tensor]):
        """Manager of the upper layer will call this function with accepted payload
        
            In synchronous mode, return True to end current FL round.
        """
        raise NotImplementedError()

    def train(self):
        """Override this method to define the training procedure. This function should manipulate :attr:`self._model`."""
        raise NotImplementedError()

    def validate(self):
        """Validate quality of local model."""
        raise NotImplementedError()

    def evaluate(self):
        """Evaluate quality of local model."""
        raise NotImplementedError()


class SerialClientTrainer(SerialModelMaintainer):
    """Base class. Simulate multiple clients in sequence in a single process.

    Args:
        model (torch.nn.Module): Model used in this federation.
        num_clients (int): Number of clients in current trainer.
        cuda (bool): Use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        personal (bool, optional): If Ture is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These paremeters are indexed by [0, num-1]. Defaults to False.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 num_clients: int,
                 cuda: bool,
                 device: str = None,
                 personal: bool = False) -> None:
        super().__init__(model, num_clients, cuda, device, personal)

        self.num_clients = num_clients
        self.dataset = FedDataset()
        self.type = SERIAL_TRAINER  # represent serial trainer

    def setup_dataset(self):
        """Override this function to set up local dataset for clients"""
        raise NotImplementedError()

    def setup_optim(self):
        """"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def uplink_package(self) -> List[List[torch.Tensor]]:
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        raise NotImplementedError()

    @abstractclassmethod
    def local_process(self, id_list: list, payload: List[torch.Tensor]):
        """Define the local main process."""
        # Args:
        #     id_list (list): The list consists of client ids.
        #     payload (List[torch.Tensor]): The information that server broadcasts to clients.
        raise NotImplementedError()

    def train(self):
        """Override this method to define the algorithm of training your model. This function should manipulate :attr:`self._model`"""
        raise NotImplementedError()

    def evaluate(self):
        """Evaluate quality of local model."""
        raise NotImplementedError()

    def validate(self):
        """Validate quality of local model."""
        raise NotImplementedError()
