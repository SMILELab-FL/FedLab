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


class SerializationTool(object):
    @staticmethod
    def serialize_model_gradients(model: torch.nn.Module, cpu:bool=True) -> torch.Tensor:
        """Vectorize model gradients.

        Args:
            model (torch.nn.Module): Model with gradients.
            cpu (bool, optional): Whether move the vectorized parameter to ``torch.device('cpu')`` by force. Defaults to ``True``. If ``cpu`` is ``False``, the returned vector is on the same device as ``model``.

        Returns:
            torch.Tensor: Vectorized model gradients. Only contains trainable parameters.
        """
        gradients = [param.grad.data.view(-1) for param in model.parameters()]
        m_gradients = torch.cat(gradients)
        if cpu:
            m_gradients = m_gradients.cpu()
        
        return m_gradients

    @staticmethod
    def deserialize_model_gradients(model: torch.nn.Module, gradients: torch.Tensor) -> None:
        """Deserialize vectorized ``gradients`` into ``model``'s ``param.grad.data`` for each trainable parameter.

        Args:
            model (torch.nn.Module): Model.
            gradients (torch.Tensor): Vectorized gradients for single model.
        """
        idx = 0
        for parameter in model.parameters():
            layer_size = parameter.grad.numel()
            shape = parameter.grad.shape

            parameter.grad.data[:] = gradients[idx:idx+layer_size].view(shape)[:]
            idx += layer_size

    @staticmethod
    def serialize_model(model: torch.nn.Module, cpu:bool=True) -> torch.Tensor:
        """Unfold model parameters, including trainable as well as untrainable parameters.
        
        Unfold every layer of model, concate all of tensors into one vector.
        Return a `torch.Tensor` with shape ``(d, )``, where ``d`` is the total number of parameters in ``model``, including trainable as well as untrainable parameters.

        Please note that we update the implementation. 
        Current version of serialization includes the parameters in batchnorm layers.

        Args:
            model (torch.nn.Module): model to serialize.
            cpu (bool, optional): Whether move the vectorized parameter to ``torch.device('cpu')`` by force. Defaults to ``True``. If ``cpu`` is ``False``, the returned vector is on the same device as ``model``.
        """
        parameters = [param.data.view(-1) for param in model.state_dict().values()]
        m_parameters = torch.cat(parameters)
        if cpu:
            m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def deserialize_model(model: torch.nn.Module,
                      serialized_parameters: torch.Tensor,
                      mode="copy"):
        """Assigns serialized parameters to parameters in ``model.state_dict()``, which includes both trainable parameters and untrainable parameters.
        This is done by iterating through ``model.state_dict()`` and assigning the relevant values with the same dimension as the ``model.state_dict()``.
        NOTE: this function manipulates ``model.state_dict()``.

        Args:
            model (torch.nn.Module): model to deserialize.
            serialized_parameters (torch.Tensor): serialized model parameters.
            mode (str): deserialize mode. Support "copy", "add", and "sub".
        """
        current_index = 0  # keep track of where to read from grad_update

        for param in model.state_dict().values():
            numel = param.numel()
            size = param.size()
            if mode == "copy":
                param.copy_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            elif mode == "add":
                param.add_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            elif mode == "sub":
                param.sub_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\", \"add\" or \"sub\" "
                    .format(mode))
            current_index += numel


    @staticmethod
    def serialize_trainable_model(model: torch.nn.Module, cpu:bool=True) -> torch.Tensor:
        """Unfold trainable model parameters.
        
        Unfold every layer of model by iterating though ``model.parameters()``,  then concate all of tensors into one vector.
        Return a `torch.Tensor` with shape (size, ).

        Args:
            model (torch.nn.Module): model to serialize.
            cpu (bool, optional): Whether move the vectorized parameter to ``torch.device('cpu')`` by force. Defaults to ``True``. If ``cpu`` is ``False``, the returned vector is on the same device as ``model``.
        """

        parameters = [param.data.view(-1) for param in model.parameters()]
        m_parameters = torch.cat(parameters)
        if cpu:
            m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def deserialize_trainable_model(model: torch.nn.Module,
                          serialized_parameters: torch.Tensor,
                          mode="copy"):
        """Assigns serialized trainable parameters to ``model.parameters``.
        This is done by iterating through ``model.parameters()`` and assigning the relevant params in ``grad_update``.
        NOTE: this function manipulates ``model.parameters()``.

        Args:
            model (torch.nn.Module): model to deserialize.
            serialized_parameters (torch.Tensor): serialized model parameters.
            mode (str): deserialize mode. Support "copy", "add", and "sub".
        """
        current_index = 0  # keep track of where to read from grad_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if mode == "copy":
                parameter.data.copy_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            elif mode == "add":
                parameter.data.add_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            elif mode == "sub":
                parameter.data.sub_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\", \"add\" or \"sub\" "
                    .format(mode))
            current_index += numel