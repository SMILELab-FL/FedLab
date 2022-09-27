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

from typing import List
import torch
from copy import deepcopy
from ..utils.serialization import SerializationTool
from ..utils.functional import get_best_gpu


class ModelMaintainer(object):
    """Maintain PyTorch model.

    Provide necessary attributes and operation methods. More features with local or global model
    will be implemented here.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool): Use GPUs or not.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest memory as default.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 cuda: bool,
                 device: str = None) -> None:
        self.cuda = cuda

        if cuda:
            # dynamic gpu acquire.
            if device is None:
                self.device = get_best_gpu()
            else:
                self.device = device
            self._model = deepcopy(model).cuda(self.device)
        else:
            self._model = deepcopy(model).cpu()

    def set_model(self, parameters: torch.Tensor):
        """Assign parameters to self._model."""
        SerializationTool.deserialize_model(self._model, parameters)

    @property
    def model(self) -> torch.nn.Module:
        """Return :class:`torch.nn.module`."""
        return self._model

    @property
    def model_parameters(self) -> torch.Tensor:
        """Return serialized model parameters."""
        return SerializationTool.serialize_model(self._model)

    @property
    def model_gradients(self) -> torch.Tensor:
        """Return serialized model gradients."""
        return SerializationTool.serialize_model_gradients(self._model)

    @property
    def shape_list(self) -> List[torch.Tensor]:
        """Return shape of model parameters.
        
        Currently, this attributes used in tensor compression.
        """
        shape_list = [param.shape for param in self._model.parameters()]
        return shape_list


class SerialModelMaintainer(ModelMaintainer):
    """"Maintain PyTorch model.

    Provide necessary attributes and operation methods. More features with local or global model
    will be implemented here.

    Args:
        model (torch.nn.Module): PyTorch model.
        num_clients (int): The number of independent models. 
        cuda (bool): Use GPUs or not.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest idle memory as default.
        personal (bool, optional): If Ture is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These paremeters are indexed by [0, num-1]. Defaults to False.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 num_clients: int,
                 cuda: bool,
                 device: str = None,
                 personal: bool = False) -> None:
        super().__init__(model, cuda, device)
        if personal:
            self.parameters = [
                self.model_parameters for _ in range(num_clients)
            ]  # A list of Tensor
        else:
            self.parameters = None

    def set_model(self, parameters: torch.Tensor = None, id: int = None):
        """Assign parameters to self._model.

        Note:
            parameters and id can not be None at the same time. 
            If id is None, this function load the given parameters.
            If id is not None, this function load the parameters of given id first and the parameters attribute will be ignored.

        Args:
            parameters (torch.Tensor, optional): Model parameters. Defaults to None.
            id (int, optional): Load the model parameters of client id. Defaults to None.
        """
        if id is None:
            super().set_model(parameters)
        else:
            super().set_model(self.parameters[id])