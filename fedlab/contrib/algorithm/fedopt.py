import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from ...utils.functional import evaluate, setup_seed, AverageMeter
from ...utils.serialization import SerializationTool
from ...utils.aggregator import Aggregators
from ...contrib.algorithm.fedavg import FedAvgSerialClientTrainer, FedAvgServerHandler


class FedOptServerHandler(FedAvgServerHandler):
    def setup_optim(self, sampler, args):
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio * self.n)
        self.round_clients = int(self.sample_ratio * self.n)
        self.sampler = sampler

        self.args = args
        self.lr = args.glr
        # momentum
        self.beta1 = self.args.beta1
        self.beta2 = self.args.beta2
        self.option = self.args.option
        self.tau = self.args.tau
        self.momentum = torch.zeros_like(self.model_parameters)
        self.vt = torch.zeros_like(self.model_parameters)
        assert self.option in ["adagrad", "yogi", "adam"]

    @property
    def num_clients_per_round(self):
        return self.round_clients

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        loss_ = AverageMeter()
        acc_ = AverageMeter()
        for id in tqdm(id_list):
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader, loss_, acc_)
            self.cache.append(pack)
        return loss_, acc_

    def global_update(self, buffer):
        gradient_list = [
            torch.sub(ele[0], self.model_parameters) for ele in buffer
        ]
        indices, _ = self.sampler.last_sampled
        delta = Aggregators.fedavg_aggregate(gradient_list,
                                             self.args.weights[indices])
        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * delta

        delta_2 = torch.pow(delta, 2)
        if self.option == "adagrad":
            self.vt += delta_2
        elif self.option == "yogi":
            self.vt = self.vt - (
                1 - self.beta2) * delta_2 * torch.sign(self.vt - delta_2)
        else:
            # adam
            self.vt = self.beta2 * self.vt + (1 - self.beta2) * delta_2

        serialized_parameters = self.model_parameters + self.lr * self.momentum / (
            torch.sqrt(self.vt) + self.tau)
        self.set_model(serialized_parameters)

