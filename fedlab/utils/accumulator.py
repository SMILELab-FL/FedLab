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


class GradientAccumulator(object):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = deepcopy(model)

        self._gradients = [torch.zeros_like(parameter.data) for parameter in self.model.parameters()]

    def reset(self, model=None):
        # reset all variables
        if model is None:
            self._gradients = [torch.zeros_like(parameter) for parameter in self._gradients]
        else:
            self.model = deepcopy(model)
            self._gradients = [torch.zeros_like(parameter.data) for parameter in self.model.parameters()]

    def delta(self, model):
        """Get the difference between incoming model and origin model.

        Returns:
            a flattern tensor.
        """
        delta = []
        for param_t, param_t0 in zip(model.parameters(), self.model.parameters()):
            delta_ = param_t.data.detach() - param_t0.data.detach()
            delta.append(delta_)
        flat_delta = [delta_.view(-1) for delta_ in delta]
        return torch.cat(flat_delta)

    def accumulate(self, model):
        """Accumulate gradients of incoming model
        
            Please ensure that `prameters.grad` is callable. 
            Therefore, this function should be called before optimizer.zero_grad().
        """
        for index, parameters in enumerate(model.parameters()):
            self._gradients[index].add_(parameters.grad.detach())

    @property
    def gradients(self):
        """Obtain the cumulative gradient"""
        flat_gradients = [grad.view(-1) for grad in self._gradients]
        return torch.cat(flat_gradients)

    @gradients.setter
    def gradients(self, flat_grad):
        """Asssign gradients"""
        pass