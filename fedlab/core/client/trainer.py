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

    def __init__(self, model, cuda):
        super().__init__(model, cuda)
        self.client_num = 1  # default is 1.
        self.dataset = FedLabDataset()
        self.type = ORDINARY_TRAINER

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
        logger (Logger, optional): Object of :class:`Logger`.
    """

    def __init__(self, model, client_num, cuda=False, logger=None):
        super().__init__(model, cuda)
        self.client_num = client_num
        self.dataset = FedLabDataset()
        self.type = SERIAL_TRAINER  # represent serial trainer
        self._LOGGER = Logger() if logger is None else logger
        self.param_list = []

    @property
    def uplink_package(self):
        return self.param_list

    # def _train_alone(self, model_parameters, train_loader):
    #     """Train local model with :attr:`model_parameters` on :attr:`train_loader`.
        
    #     Args:
    #         model_parameters (torch.Tensor): Serialized model parameters of one model.
    #         train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
    #     """
    #     raise NotImplementedError()

    # def _get_dataloader(self, client_id):
    #     """Get :class:`DataLoader` for ``client_id``."""
    #     raise NotImplementedError()

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

    # def local_process(self, id_list, payload):
    #     """Train local model with different dataset according to client id in ``id_list``.

    #     Args:
    #         id_list (list[int]): Client id in this training serial.
    #         payload (list[torch.Tensor]): communication payload from server.
    #     """
    #     self.buffer = []
    #     self._LOGGER.info(
    #         "Local training with client id list: {}".format(id_list))
    #     for idx in id_list:
    #         self._LOGGER.info(
    #             "Starting training procedure of client [{}]".format(idx))

    #         # data_loader = self._get_dataloader(client_id=idx)
    #         # self._train_alone(model_parameters=payload[0],
    #         #                   train_loader=data_loader)
    #         self.buffer.append(self.model_parameters)
    #     return self.buffer
