import threading

import torch

from fedlab.board import fedboard
from fedlab.contrib.algorithm import SyncServerHandler
from fedlab.contrib.client_sampler.uniform_sampler import RandomSampler
from fedlab.core.server import SynchronousServerManager
from fedlab.utils import MessageCode


class ExampleManager(SynchronousServerManager):
    def activate_clients(self, round):
        self._LOGGER.info("Client activation procedure")
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self._LOGGER.info("Client id list: {}".format(clients_this_round))

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            downlink_package.append(torch.tensor(round))
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(content=[id_list] + downlink_package,
                               message_code=MessageCode.ParameterUpdate,
                               dst=rank)

    def main_loop(self):
        rd = 1
        while self._handler.if_stop is not True:
            activator = threading.Thread(target=self.activate_clients, args=[rd])
            activator.start()
            total_loss = 0
            while True:
                sender_rank, message_code, payload = self._network.recv()
                if message_code == MessageCode.ParameterUpdate:
                    if self._handler.load(payload):
                        break
                total_loss += payload[1].numpy()
            metric = {'loss': total_loss, 'nloss': -total_loss}
            fedboard.log(rd, metrics=metric)
            rd += 1


class ExampleHandler(SyncServerHandler):

    def sample_clients(self, num_to_sample=None):
        if self.sampler is None:
            self.sampler = RandomSampler(self.num_clients)
        self.round_clients = max(1, int(self.sample_ratio * self.num_clients))
        sampled = self.sampler.sample(self.round_clients)
        self.round_clients = len(sampled)

        assert self.num_clients_per_round == len(sampled)
        return sorted(sampled)
