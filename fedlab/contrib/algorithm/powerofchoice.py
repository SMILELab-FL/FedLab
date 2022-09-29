import random
import numpy as np
import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDSerialClientTrainer
from ...core.standalone import StandalonePipeline
from ...utils import functional as F


#####################
#                   #
#      Pipeline     #
#                   #
#####################

class PowerofchoicePipeline(StandalonePipeline):
    def main(self):
        while self.handler.if_stop is False:
            candidates = self.handler.sample_candidates()
            losses = self.trainer.evaluate(candidates,
                                           self.handler.model_parameters)

            # server side
            sampled_clients = self.handler.sample_clients(candidates, losses)
            broadcast = self.handler.downlink_package

            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)


#####################
#                   #
#       Server      #
#                   #
#####################


class Powerofchoice(SyncServerHandler):
    def setup_optim(self, d):
        self.d = d # the number of candidate

    def sample_candidates(self):
        selection = random.sample(range(self.num_clients), self.d)
        selection = sorted(selection)
        return selection

    def sample_clients(self, candidates, losses):
        sort = np.array(losses).argsort().tolist()
        sort.reverse()
        selected_clients = np.array(candidates)[sort][0:self.num_clients_per_round]
        return selected_clients.tolist()


#####################
#                   #
#       Client      #
#                   #
#####################

class PowerofchoiceSerialClientTrainer(SGDSerialClientTrainer):
    def evaluate(self, id_list, model_parameters):
        self.set_model(model_parameters)
        losses = []
        for id in id_list:
            dataloader = self.dataset.get_dataloader(id)
            loss, acc = F.evaluate(self._model, torch.nn.CrossEntropyLoss(),
                                   dataloader)
            losses.append(loss)
        return losses