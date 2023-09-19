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
from torch import nn
import random
from copy import deepcopy

from typing import List
from ...utils import Logger, Aggregators, SerializationTool
from ...utils.functional import evaluate
from ...core.server.handler import ServerHandler
from ..client_sampler.base_sampler import FedSampler
from ..client_sampler.uniform_sampler import RandomSampler


class SyncServerHandler(ServerHandler):
    """Synchronous Parameter Server Handler.

    Backend of synchronous parameter server: this class is responsible for backend computing in synchronous server.

    Synchronous parameter server will wait for every client to finish local training process before
    the next FL round.

    Details in paper: http://proceedings.mlr.press/v54/mcmahan17a.html

    Args:
        model (torch.nn.Module): model trained by federated learning.
        global_round (int): stop condition. Shut down FL system when global round is reached.
        num_clients (int): number of clients in FL. Default: 0 (initialized external).
        sample_ratio (float): the result of ``sample_ratio * num_clients`` is the number of clients for every FL round.
        cuda (bool): use GPUs or not. Default: ``False``.
        device (str, optional): assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest memory as default.
        sampler (FedSampler, optional): assign a sampler to define the client sampling strategy. Default: random sampling with :class:`FedSampler`.
        logger (Logger, optional): object of :class:`Logger`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        num_clients: int = 0,
        sample_ratio: float = 1,
        cuda: bool = False,
        device: str = None,
        sampler: FedSampler = None,
        logger: Logger = None,
    ):
        super(SyncServerHandler, self).__init__(model, cuda, device)

        self._LOGGER = Logger() if logger is None else logger
        assert 0.0 <= sample_ratio <= 1.0

        # basic setting
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.sampler = sampler

        # client buffer
        self.round_clients = max(
            1, int(self.sample_ratio * self.num_clients)
        )  # for dynamic client sampling
        self.client_buffer_cache = []

        # stop condition
        self.global_round = global_round
        self.round = 0

    @property
    def downlink_package(self) -> List[torch.Tensor]:
        """Property for manager layer. Server manager will call this property when activates clients."""
        return [self.model_parameters]

    @property
    def num_clients_per_round(self):
        return self.round_clients

    @property
    def if_stop(self):
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return self.round >= self.global_round

    # for built-in sampler
    # @property
    # def num_clients_per_round(self):
    #     return max(1, int(self.sample_ratio * self.num_clients))

    # def setup_optim(self, num_clients):
    #     self.num_clients = num_clients

    def sample_clients(self, num_to_sample=None):
        """Return a list of client rank indices selected randomly. The client ID is from ``0`` to
        ``self.num_clients -1``."""
        # selection = random.sample(range(self.num_clients),
        #                           self.num_clients_per_round)
        # If the number of clients per round is not fixed, please change the value of self.sample_ratio correspondly.
        # self.sample_ratio = float(len(selection))/self.num_clients
        # assert self.num_clients_per_round == len(selection)

        if self.sampler is None:
            self.sampler = RandomSampler(self.num_clients)
        # new version with built-in sampler
        num_to_sample = self.round_clients if num_to_sample is None else num_to_sample
        sampled = self.sampler.sample(self.round_clients)
        self.round_clients = len(sampled)

        assert self.num_clients_per_round == len(sampled)
        return sorted(sampled)

    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

    def load(self, payload: List[torch.Tensor]) -> bool:
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
        self.client_buffer_cache.append(deepcopy(payload))

        assert len(self.client_buffer_cache) <= self.num_clients_per_round

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1

            # reset cache
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False

    def setup_dataset(self, dataset) -> None:
        self.dataset = dataset

    def evaluate(self):
        self._model.eval()
        test_loader = self.dataset.get_dataloader(type="test", batch_size=128)
        loss_, acc_ = evaluate(self._model, nn.CrossEntropyLoss(), test_loader)
        self._LOGGER.info(
            f"Round [{self.round - 1}/{self.global_round}] test performance on server: \t Loss: {loss_:.5f} \t Acc: {100*acc_:.3f}%"
        )

        return loss_, acc_


class AsyncServerHandler(ServerHandler):
    """Asynchronous Parameter Server Handler

    Update global model immediately after receiving a ParameterUpdate message
    Paper: https://arxiv.org/abs/1903.03934

    Args:
        model (torch.nn.Module): Global model in server
        global_round (int): stop condition. Shut down FL system when global round is reached.
        num_clients (int): number of clients in FL.
        cuda (bool): Use GPUs or not.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest memory as default.
        logger (Logger, optional): Object of :class:`Logger`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        num_clients: int,
        cuda: bool = False,
        device: str = None,
        logger: Logger = None,
    ):
        super(AsyncServerHandler, self).__init__(model, cuda, device)
        self._LOGGER = Logger() if logger is None else logger
        self.num_clients = num_clients
        self.round = 0
        self.global_round = global_round

    @property
    def if_stop(self):
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return self.round >= self.global_round

    @property
    def downlink_package(self):
        return [self.model_parameters, torch.Tensor([self.round])]

    def setup_optim(self, alpha, strategy="constant", a=10, b=4):
        """Setup optimization configuration.

        Args:
            alpha (float): Weight used in async aggregation.
            strategy (str, optional): Adaptive strategy. ``constant``, ``hinge`` and ``polynomial`` is optional. Default: ``constant``.. Defaults to 'constant'.
            a (int, optional): Parameter used in async aggregation.. Defaults to 10.
            b (int, optional): Parameter used in async aggregation.. Defaults to 4.
        """
        # async aggregation params
        self.alpha = alpha
        self.strategy = strategy  # "constant", "hinge", "polynomial"
        self.a = a
        self.b = b

    def global_update(self, buffer):
        client_model_parameters, model_time = buffer[0], buffer[1].item()
        """ "update global model from client_model_queue"""
        alpha_T = self.adapt_alpha(model_time)
        aggregated_params = Aggregators.fedasync_aggregate(
            self.model_parameters, client_model_parameters, alpha_T
        )  # use aggregator
        SerializationTool.deserialize_model(self._model, aggregated_params)

    def load(self, payload: List[torch.Tensor]) -> bool:
        self.global_update(payload)
        self.round += 1

    def adapt_alpha(self, receive_model_time):
        """update the alpha according to staleness"""
        staleness = self.round - receive_model_time
        if self.strategy == "constant":
            return torch.mul(self.alpha, 1)
        elif self.strategy == "hinge" and self.b is not None and self.a is not None:
            if staleness <= self.b:
                return torch.mul(self.alpha, 1)
            else:
                return torch.mul(self.alpha, 1 / (self.a * ((staleness - self.b) + 1)))
        elif self.strategy == "polynomial" and self.a is not None:
            return torch.mul(self.alpha, (staleness + 1) ** (-self.a))
        else:
            raise ValueError("Invalid strategy {}".format(self.strategy))
