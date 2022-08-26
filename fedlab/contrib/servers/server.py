

import torch
import random
from copy import deepcopy

from ...utils import Logger, Aggregators, SerializationTool
from ...core.server.handler import ServerHandler

class SyncServerHandler(ServerHandler):
    """Synchronous Parameter Server Handler.

    Backend of synchronous parameter server: this class is responsible for backend computing in synchronous server.

    Synchronous parameter server will wait for every client to finish local training process before
    the next FL round.

    Details in paper: http://proceedings.mlr.press/v54/mcmahan17a.html

    Args:
        model (torch.nn.Module): Model used in this federation.
        global_round (int): stop condition. Shut down FL system when global round is reached.
        sample_ratio (float): The result of ``sample_ratio * client_num`` is the number of clients for every FL round.
        cuda (bool): Use GPUs or not. Default: ``False``.
        logger (Logger, optional): object of :class:`Logger`.
    """

    def __init__(self,
                 model,
                 global_round,
                 sample_ratio,
                 cuda=False,
                 logger=None):
        super(SyncServerHandler, self).__init__(model, cuda)

        self._LOGGER = Logger() if logger is None else logger

        assert sample_ratio >= 0.0 and sample_ratio <= 1.0

        # basic setting
        self.client_num = 0
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
        return max(1, int(self.sample_ratio * self.client_num))

    def sample_clients(self):
        """Return a list of client rank indices selected randomly. The client ID is from ``0`` to
        ``self.client_num -1``."""
        selection = random.sample(range(self.client_num),
                                  self.client_num_per_round)
        return selection

    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

    def load(self, payload):
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
            self.client_buffer_cache.append(deepcopy(payload))
        else:
            self.client_buffer_cache += payload  # serial trainer

        assert len(self.client_buffer_cache) <= self.client_num_per_round
        
        if len(self.client_buffer_cache) == self.client_num_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1

            # reset cache
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False


class AsyncServerHandler(ServerHandler):
    """Asynchronous Parameter Server Handler

    Update global model immediately after receiving a ParameterUpdate message
    Paper: https://arxiv.org/abs/1903.03934

    Args:
        model (torch.nn.Module): Global model in server
        alpha (float): Weight used in async aggregation.
        global_round (int): Stop condition. Shut down FL system when global_round is reached.
        strategy (str): Adaptive strategy. ``constant``, ``hinge`` and ``polynomial`` is optional. Default: ``constant``.
        cuda (bool): Use GPUs or not.
        logger (Logger, optional): Object of :class:`Logger`.
    """
    def __init__(self,
                 model,
                 alpha,
                 global_round,
                 strategy="constant",
                 cuda=False,
                 logger=None):
        super(AsyncServerHandler, self).__init__(model, cuda)
        self._LOGGER = Logger() if logger is None else logger

        self.client_num = 0

        self.round = 1
        self.global_round = global_round

        # async aggregation params
        self.alpha = alpha
        self.strategy = strategy  # "constant", "hinge", "polynomial"
        self.a = 10
        self.b = 4

    @property
    def if_stop(self):
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return self.round >= self.global_round

    @property
    def downlink_package(self):
        return [self.model_parameters, torch.Tensor([self.round])]

    def global_update(self, buffer):
        client_model_parameters, model_time = buffer[0], buffer[1].item()
        """ "update global model from client_model_queue"""
        alpha_T = self._adapt_alpha(model_time)
        aggregated_params = Aggregators.fedasync_aggregate(
            self.model_parameters, client_model_parameters,
            alpha_T)  # use aggregator
        SerializationTool.deserialize_model(self._model, aggregated_params)

    def load(self, payload):
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
                return torch.mul(self.alpha,
                                 1 / (self.a * ((staleness - self.b) + 1)))
        elif self.strategy == "polynomial" and self.a is not None:
            return torch.mul(self.alpha, (staleness + 1)**(-self.a))
        else:
            raise ValueError("Invalid strategy {}".format(self.strategy))
