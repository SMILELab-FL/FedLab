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

import torch
from copy import deepcopy
from ..utils.serialization import SerializationTool


class ModelMaintainer(object):
    """Maintain PyTorch model.

        Provide necessary attributes and operation methods.

    Args:
        model (torch.Module): PyTorch model.
        cuda (bool): use GPUs or not.
    """
    def __init__(self, model, cuda) -> None:
        
        self.cuda = cuda

        if cuda:
            self._model = model.cuda()
        else:
            self._model = model.cpu()

    @property
    def model(self):
        """Return torch.nn.module"""
        return self._model

    @property
    def model_parameters(self):
        """Return serialized model parameters."""
        return SerializationTool.serialize_model(self._model)

    @property
    def shape_list(self):
        """attribute"""
        shape_list = [param.shape for param in self._model.parameters()]
        return shape_list
