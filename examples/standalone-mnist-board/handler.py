from copy import deepcopy
from typing import List

import torch

from fedlab.contrib.client_sampler.base_sampler import FedSampler
from fedlab.contrib.client_sampler.uniform_sampler import RandomSampler
from fedlab.core.server.handler import ServerHandler
from fedlab.utils import Logger, Aggregators, SerializationTool


class StandaloneSyncServerHandler(ServerHandler):

    def __init__(
            self,
            model: torch.nn.Module,
            global_round: int,
            sample_ratio: float,
            cuda: bool = False,
            device: str = None,
            sampler: FedSampler = None,
            logger: Logger = None,
    ):
        super(StandaloneSyncServerHandler, self).__init__(model, cuda, device)

        self._LOGGER = Logger() if logger is None else logger
        assert sample_ratio >= 0.0 and sample_ratio <= 1.0

        # basic setting
        self.num_clients = 0
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
        sampled = self.sampler.sample(num_to_sample)
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
