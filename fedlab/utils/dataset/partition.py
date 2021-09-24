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

import warnings
from abc import ABC, abstractmethod

import numpy as np

import torch
import torchvision

from . import functional as F


class DataPartitioner(ABC):

    @abstractmethod
    def _perform_partition(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class CIFAR10Partitioner(DataPartitioner):
    """CIFAR10 data partitioner.

    Partition CIFAR10 given specific client number. Currently 6 supported partition schemes can be
    achieved by passing different combination of parameters in initialization:

    - ``balance=None``

      - ``partition="dirichlet"``: non-iid partition used in
        `Bayesian Nonparametric Federated Learning of Neural Networks <https://arxiv.org/abs/1905.12022>`_
        and `Federated Learning with Matched Averaging <https://arxiv.org/abs/2002.06440>`_. Refer
        to :func:`fedlab.utils.dataset.functional.hetero_dir_partition` for more information.

      - ``partition="shards"``: non-iid method used in FedAvg `paper <https://arxiv.org/abs/1602.05629>`_.
        Refer to :func:`fedlab.utils.dataset.functional.shards_partition` for more information.


    - ``balance=True``: "Balance" refers to FL scenario that sample numbers for different clients
      are the same. Refer to :func:`fedlab.utils.dataset.functional.balance_partition` for more
      information.

      - ``partition="iid"``: Random select samples from complete dataset given sample number for
        each client.

      - ``partition="dirichlet"``: Refer to :func:`fedlab.utils.dataset.functional.client_inner_dirichlet_partition`
        for more information.

    - ``balance=False``: "Unbalance" refers to FL scenario that sample numbers for different clients
      are different. For unbalance method, sample number for each client is drown from Log-Normal
      distribution with variance ``unbalanced_sgm``. When ``unbalanced_sgm=0``, partition is
      balanced. Refer to :func:`fedlab.utils.dataset.functional.lognormal_unbalance_partition`
      for more information. The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.

      - ``partition="iid"``: Random select samples from complete dataset given sample number for
        each client.

      - ``partition="dirichlet"``: Refer to :func:`fedlab.utils.dataset.functional.client_inner_dirichlet_partition`
        for more information.

    Args:
        targets (list or numpy.ndarray): Targets of dataset for partition. Each element is in range of [0, 1, ..., 9].
        num_clients (int): Number of clients for data partition.
        balance (bool, optional): Balanced partition over all clients or not. Default as ``True``.
        partition (str, optional): Partition type, only ``"iid"``, ``shards``, ``"dirichlet"`` are supported. Default as ``"iid"``.
        unbalance_sgm (float, optional): Log-normal distribution variance for unbalanced data partition over clients. Default as ``0`` for balanced partition.
        num_shards (int, optional): Number of shards in non-iid ``"shards"`` partition. Only works if ``partition="shards"``. Default as ``None``.
        dir_alpha (float, optional): Dirichlet distribution parameter for non-iid partition. Only works if ``partition="dirichlet"``. Default as ``None``.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.
        seed (int, optional): Random seed. Default as ``None``.
    """

    def __init__(self, targets, num_clients,
                 balance=True, partition="iid",
                 unbalance_sgm=0,
                 num_shards=None,
                 dir_alpha=None,
                 verbose=True,
                 seed=None):

        self.targets = np.array(targets)  # with shape (num_samples,)
        self.num_samples = self.targets.shape[0]
        self.num_classes = 10
        self.num_clients = num_clients
        self.client_dict = dict()
        self.partition = partition
        self.balance = balance
        self.dir_alpha = dir_alpha
        self.num_shards = num_shards
        self.unbalance_sgm = unbalance_sgm
        self.verbose = verbose
        # self.rng = np.random.default_rng(seed)  # rng currently not supports randint
        np.random.seed(seed)

        # partition scheme check
        if balance is None:
            assert partition in ["dirichlet", "shards"], f"When balance=None, 'partition' only " \
                                                         f"accepts 'dirichlet' and 'shards'."
        elif isinstance(balance, bool):
            assert partition in ["iid", "dirichlet"], f"When balance is bool, 'partition' only " \
                                                      f"accepts 'dirichlet' and 'iid'."
        else:
            raise ValueError(f"'balance' can only be NoneType or bool, not {type(balance)}.")

        # perform partition according to setting
        self.client_dict = self._perform_partition()
        # get sample number count for each client
        self.client_sample_count = F.samples_num_count(self.client_dict, self.num_clients)

    def _perform_partition(self):
        if self.balance is None:
            if self.partition == "dirichlet":
                client_dict = F.hetero_dir_partition(self.targets,
                                                     self.num_clients,
                                                     self.num_classes,
                                                     self.dir_alpha,
                                                     min_require_size=self.num_classes)

            else:  # partition is 'shards'
                client_dict = F.shards_partition(self.targets, self.num_clients, self.num_shards)

        else:  # if balance is True or False
            # perform sample number balance/unbalance partition over all clients
            if self.balance is True:
                client_sample_nums = F.balance_partition(self.num_clients, self.num_samples)
            else:
                client_sample_nums = F.lognormal_unbalance_partition(self.num_clients,
                                                                     self.num_samples,
                                                                     self.unbalance_sgm)

            # perform iid/dirichlet partition for each client
            rand_perm = np.random.permutation(self.num_samples)
            if self.partition == "iid":
                num_cumsum = np.cumsum(client_sample_nums).astype(int)
                client_dict = F.split_indices(num_cumsum, rand_perm)
            else:  # for dirichlet
                targets = self.targets[rand_perm]
                client_dict = F.client_inner_dirichlet_partition(targets, self.num_clients,
                                                                 self.num_classes, self.dir_alpha,
                                                                 client_sample_nums, self.verbose)

        return client_dict

    def __getitem__(self, index):
        """Obtain sample indices for client ``index``.

        Args:
            index (int): Client ID.

        Returns:
            list: List of sample indices for client ID ``index``.

        """
        return self.client_dict[index]

    def __len__(self):
        """Usually equals to number of clients."""
        return len(self.client_dict)
