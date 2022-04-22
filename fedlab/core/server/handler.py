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
import random
import torch

from ..model_maintainer import ModelMaintainer
from ...utils import Logger, Aggregators, SerializationTool


class ParameterServerBackendHandler(ModelMaintainer):
    """An abstract class representing handler of parameter server.

    Please make sure that your self-defined server handler class subclasses this class

    Example:
        Read source code of :class:`SyncParameterServerHandler` and :class:`AsyncParameterServerHandler`.
    """

    def __init__(self, model, cuda=False):
        super().__init__(model, cuda)

    @abstractproperty
    def downlink_package(self):
        """Property for manager layer. Server manager will call this property when activates clients."""
        raise NotImplementedError()

    @abstractproperty
    def if_stop(self):
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return False

    def _update_global_model(self, *args, **kwargs):
        """Override this function for iterating global model (aggregation or optimization)."""
        raise NotImplementedError()


class SyncParameterServerHandler(ParameterServerBackendHandler):
    """Synchronous Parameter Server Handler.

    Backend of synchronous parameter server: this class is responsible for backend computing in synchronous server.

    Synchronous parameter server will wait for every client to finish local training process before
    the next FL round.

    Details in paper: http://proceedings.mlr.press/v54/mcmahan17a.html

    Args:
        model (torch.nn.Module): Model used in this federation.
        global_round (int): stop condition. Shut down FL system when global round is reached.
        cuda (bool): Use GPUs or not. Default: ``False``
        sample_ratio (float): ``sample_ratio * client_num`` is the number of clients to join in every FL round. Default: ``1.0``.
        logger (Logger, optional): object of :class:`Logger`.
    """

    def __init__(self,
                 model,
                 global_round,
                 sample_ratio,
                 cuda=False,
                 logger=None):
        super(SyncParameterServerHandler, self).__init__(model, cuda)

        self._LOGGER = Logger() if logger is None else logger

        assert sample_ratio >= 0.0 and sample_ratio <= 1.0

        # basic setting
        self.client_num_in_total = 0
        self.sample_ratio = sample_ratio

        # client buffer
        self.client_buffer_cache = []

        # stop condition
        self.global_round = global_round
        self.round = 0

    @property
    def downlink_package(self):
        """Property for manager layer. Server manager will call this property when activates clients."""
        return [self.model_parameters]

    @property
    def if_stop(self):
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return self.round >= self.global_round

    @property
    def client_num_per_round(self):
        return max(1, int(self.sample_ratio * self.client_num_in_total))

    def sample_clients(self):
        """Return a list of client rank indices selected randomly. The client ID is from ``1`` to
        ``self.client_num_in_total + 1``."""
        selection = random.sample(range(self.client_num_in_total),
                                  self.client_num_per_round)
        return selection

    def _update_global_model(self, payload):
        """Update global model with collected parameters from clients.

        Note:
            Server handler will call this method when its ``client_buffer_cache`` is full. User can
            overwrite the strategy of aggregation to apply on :attr:`model_parameters_list`, and
            use :meth:`SerializationTool.deserialize_model` to load serialized parameters after
            aggregation into :attr:`self._model`.

        Args:
            payload (list[torch.Tensor]): A list of tensors passed by manager layer.
        """
        assert len(payload) > 0

        if len(payload) == 1:
            self.client_buffer_cache.append(payload[0].clone())
        else:
            self.client_buffer_cache += payload  # serial trainer

        assert len(self.client_buffer_cache) <= self.client_num_per_round
        
        if len(self.client_buffer_cache) == self.client_num_per_round:
            model_parameters_list = self.client_buffer_cache
            # use aggregator
            serialized_parameters = Aggregators.fedavg_aggregate(
                model_parameters_list)
            SerializationTool.deserialize_model(self._model, serialized_parameters)
            self.round += 1

            # reset cache cnt
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False


class AsyncParameterServerHandler(ParameterServerBackendHandler):
    """Asynchronous Parameter Server Handler

    Update global model immediately after receiving a ParameterUpdate message
    Paper: https://arxiv.org/abs/1903.03934

    Args:
        model (torch.nn.Module): Global model in server
        alpha (float): Weight used in async aggregation.
        total_time (int): Stop condition. Shut down FL system when total_time is reached.
        strategy (str): Adaptive strategy. ``constant``, ``hinge`` and ``polynomial`` is optional. Default: ``constant``.
        cuda (bool): Use GPUs or not.
        logger (Logger, optional): Object of :class:`Logger`.
    """
    def __init__(self,
                 model,
                 alpha,
                 total_time,
                 strategy="constant",
                 cuda=False,
                 logger=None):
        super(AsyncParameterServerHandler, self).__init__(model, cuda)
        self._LOGGER = Logger() if logger is None else logger

        self.client_num_in_total = 0

        self.time = 1
        self.total_time = total_time

        # async aggregation params
        self.alpha = alpha
        self.strategy = strategy  # "constant", "hinge", "polynomial"
        self.a = 10
        self.b = 4

    @property
    def if_stop(self):
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return self.time >= self.total_time

    @property
    def downlink_package(self):
        return [self.model_parameters, torch.Tensor([self.time])]

    def _update_global_model(self, payload):
        client_model_parameters, model_time = payload[0], payload[1].item()
        """ "update global model from client_model_queue"""
        alpha_T = self._adapt_alpha(model_time)
        aggregated_params = Aggregators.fedasync_aggregate(
            self.model_parameters, client_model_parameters,
            alpha_T)  # use aggregator
        SerializationTool.deserialize_model(self._model, aggregated_params)
        self.time += 1

    def _adapt_alpha(self, receive_model_time):
        """update the alpha according to staleness"""
        staleness = self.time - receive_model_time
        if self.strategy == "constant":
            return torch.mul(self.alpha, 1)
        elif self.strategy == "hinge" and self.b is not None and self.a is not None:
            if staleness <= self.b:
                return torch.mul(self.alpha, 1)
            else:
                return torch.mul(self.alpha,
                                 1 / (self.a * ((staleness - self.b) + 1)))
        elif self.strategy == "polynomial" and self.a is not None:
            return torch.mul(self.alpha, (staleness + 1)**(-self.a))
        else:
            raise ValueError("Invalid strategy {}".format(self.strategy))
