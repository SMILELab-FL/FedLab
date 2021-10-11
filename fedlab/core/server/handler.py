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

import random
import torch

from ...utils.serialization import SerializationTool
from ...utils.aggregator import Aggregators
from ...utils import Logger
from ..model_maintainer import ModelMaintainer


class ParameterServerBackendHandler(ModelMaintainer):
    """An abstract class representing handler of parameter server.

    Please make sure that your self-defined server handler class subclasses this class

    Example:
        Read source code of :class:`SyncParameterServerHandler` and :class:`AsyncParameterServerHandler`.
    """

    def __init__(self, model, cuda=False):
        super().__init__(model, cuda)

    def _update_model(self, model_parameters_list):
        """Override this function for global model aggregation strategy.

        Args:
            model_parameters_list (list[torch.Tensor]): A list of serialized model parameters collected from different clients.
        """
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
                 global_round=5,
                 cuda=False,
                 sample_ratio=1.0,
                 logger=Logger()):
        super(SyncParameterServerHandler, self).__init__(model, cuda)

        self._LOGGER = logger

        if sample_ratio < 0.0 or sample_ratio > 1.0:
            raise ValueError("Invalid select ratio: {}".format(sample_ratio))

        # basic setting
        self.client_num_in_total = 0
        self.sample_ratio = sample_ratio

        # client buffer
        self.client_buffer_cache = []
        self.cache_cnt = 0

        # stop condition
        self.global_round = global_round
        self.round = 0

    def stop_condition(self) -> bool:
        """:class:`NetworkManager` keeps monitoring the return of this method, and it will stop all related processes and threads when ``True`` returned."""
        return self.round >= self.global_round

    def sample_clients(self):
        """Return a list of client rank indices selected randomly. The client ID is from ``1`` to
        ``self.client_num_in_total + 1``."""
        selection = random.sample(range(self.client_num_in_total),
                                  self.client_num_per_round)
        return selection

    def add_model(self, sender_rank, model_parameters):
        """Deal with incoming model parameters from one client.

        Note:
            Return True when self._update_model is called.

        Args:
            sender_rank (int): Rank of sender client in ``torch.distributed`` group.
            model_parameters (torch.Tensor): Serialized model parameters from one client.
        """

        self.client_buffer_cache.append(model_parameters.clone())
        self.cache_cnt += 1

        # cache is full
        if self.cache_cnt == self.client_num_per_round:
            self._update_model(self.client_buffer_cache)
            self.round += 1
            return True
        else:
            return False

    def _update_model(self, model_parameters_list):
        """Update global model with collected parameters from clients.

        Note:
            Server handler will call this method when its ``client_buffer_cache`` is full. User can
            overwrite the strategy of aggregation to apply on :attr:`model_parameters_list`, and
            use :meth:`SerializationTool.deserialize_model` to load serialized parameters after
            aggregation into :attr:`self._model`.

        Args:
            model_parameters_list (list[torch.Tensor]): A list of parameters.aq
        """
        self._LOGGER.info(
            "Model parameters aggregation, number of aggregation elements {}".
                format(len(model_parameters_list)))
        # use aggregator
        serialized_parameters = Aggregators.fedavg_aggregate(
            model_parameters_list)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

        # reset cache cnt
        self.cache_cnt = 0
        self.client_buffer_cache = []
        self.train_flag = False

    @property
    def client_num_per_round(self):
        return max(1, int(self.sample_ratio * self.client_num_in_total))


class AsyncParameterServerHandler(ParameterServerBackendHandler):
    """Asynchronous Parameter Server Handler

    Update global model immediately after receiving a ParameterUpdate message
    Paper: https://arxiv.org/abs/1903.03934

    Args:
        model (torch.nn.Module): Global model in server
        alpha (float): weight used in async aggregation.
        total_time (int): stop condition. Shut down FL system when total_time is reached.
        strategy (str): adaptive strategy. ``constant``, ``hinge`` and ``polynomial`` is optional. Default: ``constant``.
        cuda (bool): Use GPUs or not.
        logger (Logger, optional): object of :class:`Logger`.
    """

    def __init__(self,
                 model,
                 alpha=0.5,
                 total_time=5,
                 strategy="constant",
                 cuda=False,
                 logger=Logger()):
        super(AsyncParameterServerHandler, self).__init__(model, cuda)
        self._LOGGER = logger

        self.client_num_in_total = 0

        self.current_time = 1
        self.total_time = total_time

        # async aggregation params
        self.alpha = alpha
        self.strategy = strategy  # "constant", "hinge", "polynomial"
        self.a = None
        self.b = None

    @property
    def server_time(self):
        return self.current_time

    def stop_condition(self) -> bool:
        """:class:`NetworkManager` keeps monitoring the return of this method,
        and it will stop all related processes and threads when ``True`` returned."""
        return self.current_time >= self.total_time

    def _update_model(self, client_model_parameters, model_time):
        """ "update global model from client_model_queue"""
        alpha_T = self._adapt_alpha(model_time)
        aggregated_params = Aggregators.fedasync_aggregate(
            self.model_parameters, client_model_parameters,
            alpha_T)  # use aggregator
        SerializationTool.deserialize_model(self._model, aggregated_params)
        self.current_time += 1

    def _adapt_alpha(self, receive_model_time):
        """update the alpha according to staleness"""
        staleness = self.current_time - receive_model_time
        if self.strategy == "constant":
            return torch.mul(self.alpha, 1)
        elif self.strategy == "hinge" and self.b is not None and self.a is not None:
            if staleness <= self.b:
                return torch.mul(self.alpha, 1)
            else:
                return torch.mul(self.alpha,
                                 1 / (self.a * ((staleness - self.b) + 1)))
        elif self.strategy == "polynomial" and self.a is not None:
            return (staleness + 1) ** (-self.a)
        else:
            raise ValueError("Invalid strategy {}".format(self.strategy))
