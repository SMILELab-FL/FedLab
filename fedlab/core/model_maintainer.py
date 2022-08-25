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


from ..utils.serialization import SerializationTool
from ..utils.functional import get_best_gpu


class ModelMaintainer(object):
    """Maintain PyTorch model.

    Provide necessary attributes and operation methods. More features with local or global model
    will be implemented here.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool): use GPUs or not.
        device (str): cuda device.
    """
    def __init__(self, model, cuda, device=None) -> None:
        self.cuda = cuda

        if cuda:
            # dynamic gpu acquire.
            if device is None:
                self.device = get_best_gpu()
            else:
                self.device = device
            self._model = model.cuda(self.device)
        else:
            self._model = model.cpu()

    def set_model(self, parameters):
        """Assign parameters to self._model"""
        SerializationTool.deserialize_model(self._model, parameters)

    @property
    def model(self):
        """Return :class:`torch.nn.module`"""
        return self._model

    @property
    def model_parameters(self):
        """Return serialized model parameters."""
        return SerializationTool.serialize_model(self._model)

    @property
    def model_gradients(self):
        """Return serialized model gradients."""
        return SerializationTool.serialize_model_gradients(self._model)

    @property
    def shape_list(self):
        """Return shape of model parameters.
        
        Currently, this attributes used in tensor compression.
        """
        shape_list = [param.shape for param in self._model.parameters()]
        return shape_list


class SerialModelMaintainer(ModelMaintainer):
    """Maintain PyTorch model.

    Provide necessary attributes and operation methods. More features with local or global model
    will be implemented here.

    Args:
        model (torch.nn.Module): PyTorch model.
        num (int): the number of duplicate of model parameters.
        cuda (bool): use GPUs or not.
        device (str): cuda device.
    """
    def __init__(self, model, num, cuda, device=None, personal=False) -> None:
        super().__init__(model, cuda, device)
        if personal:
            self.parameters = [self.model_parameters for _ in range(num)] # A list of Tensor
        else:
            self.parameters = None
        
    def set_model(self, parameters, id=None):
        """Assign parameters to self._model"""
        if id is None:
            super().set_model(parameters)
        else:
            super().set_model(self.parameters[id])