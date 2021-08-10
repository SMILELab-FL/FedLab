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

import logging
import random
import torch

from abc import ABC, abstractmethod
from ...utils.serialization import SerializationTool
from ...utils.aggregator import Aggregators
from ...utils.logger import Logger


class ParameterServerBackendHandler(ABC):
    """An abstract class representing handler of parameter server.

    Please make sure that your self-defined server handler class subclasses this class

    Example:
        Read source code of :class:`SyncParameterServerHandler` and :class:`AsyncParameterServerHandler`.
    """

    def __init__(self, model: torch.nn.Module, cuda=False) -> None:
        self.cuda = cuda
        if cuda:
            self._model = model.cuda()
        else:
            self._model = model.cpu()

    @abstractmethod
    def _update_model(self, serialized_params_list) -> torch.Tensor:
        """Override this function to update global model

        Args:
            serialized_params_list (list[torch.Tensor]): A list of serialized model parameters collected from different clients.
        """
        raise NotImplementedError()

    @abstractmethod
    def stop_condition(self) -> bool:
        """Override this function to tell up layer when to stop process.

        :class:`NetworkManager` keeps monitoring the return of this method, and it will stop all related processes and threads when ``True`` returned.
        """
        raise NotImplementedError()

    @property
    def model(self):
        return self._model

    @property
    def model_parameters(self):
        """Return serialized model parameters."""
        return SerializationTool.serialize_model(self._model)


class SyncParameterServerHandler(ParameterServerBackendHandler):
    """Synchronous Parameter Server Handler

    Backend of synchronous parameter server: this class is responsible for backend computing in synchronous server.

    Synchronous parameter server will wait for every client to finish local training process before
    the next FL round.

    Details in paper: http://proceedings.mlr.press/v54/mcmahan17a.html

    Args:
        model (torch.nn.Module): Model used in this federation
        client_num_in_total (int): Total number of clients in this federation
        cuda (bool): Use GPUs or not. Default: ``False``
        sample_ratio (float): ``sample_ratio * client_num`` is the number of clients to join in every FL round. Default: ``1.0``.
        logger (Logger, optional): :attr:`logger` for server handler. If set to ``None``, none logging output files will be generated while only on screen. Default: ``None``.
    """

    def __init__(
        self,
        model,
        client_num_in_total,
        global_round=1,
        cuda=False,
        sample_ratio=1.0,
        logger=None,
    ):
        super(SyncParameterServerHandler, self).__init__(model, cuda)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        if sample_ratio < 0.0 or sample_ratio > 1.0:
            raise ValueError("Invalid select ratio: {}".format(sample_ratio))

        if client_num_in_total < 1:
            raise ValueError(
                "Invalid total client number: {}".format(client_num_in_total)
            )

        # basic setting
        self.client_num_in_total = client_num_in_total
        self.sample_ratio = sample_ratio
        self.client_num_per_round = max(
            1, int(self.sample_ratio * self.client_num_in_total)
        )

        # client buffer
        self.client_buffer_cache = {}
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
        selection = random.sample(
            range(1, self.client_num_in_total + 1), self.client_num_per_round
        )
        return selection

    def add_model(self, sender_rank, serialized_params):
        """Deal with incoming model parameters from one client.

        Args:
            sender_rank (int): Rank of sender client in ``torch.distributed`` group.
            serialized_params (torch.Tensor): Serialized model parameters from one client.
        """
        # TODO: Reconsider what to return for different condition, and whether to decouple the
        #  update of single client model cache and update of global model? Not done yet
        if self.client_buffer_cache.get(sender_rank) is not None:
            self._LOGGER.info(
                "parameters from {} have already existed".format(sender_rank)
            )
            return

        self.cache_cnt += 1
        self.client_buffer_cache[sender_rank] = serialized_params.clone()

        # cache is full.
        if self.cache_cnt == self.client_num_per_round:
            self._update_model(list(self.client_buffer_cache.values()))
            self.round += 1
            return True
        else:
            return False

    def _update_model(self, serialized_params_list):
        """Update global model with collected parameters from clients.

        Note:
            Server handler will call this method when its ``client_buffer_cache`` is full. User can
            overwrite the strategy of aggregation to apply on :attr:`serialized_params_list`, and
            use :meth:`SerializationTool.deserialize_model` to load serialized parameters after
            aggregation into :attr:`self._model`.

        Args:
            serialized_params_list (list[torch.Tensor]): A list of parameters.
        """
        # use aggregator
        serialized_parameters = Aggregators.fedavg_aggregate(serialized_params_list)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

        # reset
        self.cache_cnt = 0
        self.client_buffer_cache = {}
        self.train_flag = False


class AsyncParameterServerHandler(ParameterServerBackendHandler):
    """Asynchronous Parameter Server Handler

    Update global model immediately after receiving a ParameterUpdate message
    Paper: https://arxiv.org/abs/1903.03934

    Args:
        model (torch.nn.Module): Global model in server
        client_num_in_total (int): Total number of clients in federation.
        cuda (bool): Use GPUs or not.
        logger (Logger, optional): :attr:`logger` for server handler. If set to ``None``, none logging output files will be generated while only on screen. Default: ``None``.
    """

    def __init__(self, model, client_num_in_total, cuda=False, logger=None):
        super(AsyncParameterServerHandler, self).__init__(model, cuda)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        self.alpha = 0.5
        self.client_num_in_total = client_num_in_total

        self.global_time = 5
        self.current_time = torch.zeros(1)

    def _update_model(self, model_parameters, model_time):
        """ "update global model from client_model_queue"""
        latest_serialized_parameters = self.model_parameters
        merged_params = Aggregators.fedasgd_aggregate(
            latest_serialized_parameters, model_parameters, self.alpha
        )  # use aggregator
        SerializationTool.deserialize_model(self._model, merged_params)
        self.current_time += 1

    def stop_condition(self) -> bool:
        """:class:`NetworkManager` keeps monitoring the return of this method, and it will stop all related processes and threads when ``True`` returned."""
        return self.current_time >= self.global_time

    def adapt_alpha(self, receive_model_time):
        """update the alpha according to staleness"""
        self.alpha = torch.mul(self.alpha, 1)
