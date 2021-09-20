# -*- coding: utf-8 -*-
# @Time    : 9/20/21 3:06 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : partition.py
# @Software: PyCharm

import warnings
import numpy as np

import torch
import torchvision


class DataPartitioner(object):
    def partition(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CIFAR10Partitioner(DataPartitioner):
    """CIFAR10 data partitioner.

    Partition CIFAR10 for specific client number. Current supported partition schemes: balanced iid,
    unbalanced iid, balanced non-iid, unbalanced non-iid.

    _Balance_ refers to that sample numbers for different clients are the same. For unbalance
    method, sample number for each client is drown from Log-Normal distribution with variance
    ``unbalanced_sgm``. When ``unbalanced_sgm=0``, partition is balanced. For more information,
    please refer to paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.

    For iid and non-iid, we use random sampling for iid, and use ``"shards"`` method or
    ``"dirichlet"`` method for non-iid partition. ``"shards"`` is non-iid method used in FedAvg
    `paper <https://arxiv.org/abs/1602.05629>`_. ``"dirichlet"`` is non-iid method used in
    `Federated Learning with Matched Averaging <https://arxiv.org/abs/2002.06440>`_ and
    `Bayesian nonparametric federated learning of neural networks <https://arxiv.org/abs/1905.12022>`_.




    Args:
        targets (list or numpy.ndarray): Targets of dataset for partition. Each element is in range of [0, 1, ..., 9].
        num_clients (int): Number of clients for data partition.
        partition (str, optional): Partition type, only ``"iid"`` or ``niid`` supported. Default as ``"iid"``.
        num_shards (int optional): Number of shards in non-iid partition. Only works if ``partition="niid"`` and ``dirichlet=None``. Default as ``None``.
        dirichlet (float, optional): Dirichlet distribution parameter for non-iid partition. Only works if ``partition="niid"`` and ``num_shards=None``. Default as ``None``.
        balance (bool, optional): Balanced partition over all clients or not. Default as ``True``.
        unbalance_sgm (float, optional): Log-normal distribution variance for unbalanced data partition over clients. Default as ``0`` for balanced partition.
    """

    def __init__(self, targets, num_clients,
                 partition="iid",
                 niid_method=None,
                 num_shards=None,
                 dirichlet=None,
                 balance=True,
                 unbalance_sgm=0,
                 seed=None):
        self.targets = np.array(targets)  # with shape (num_samples,)
        self.num_samples = self.targets.shape[0]
        self.num_classes = 10
        self.num_clients = num_clients
        self.client_dict = dict()
        self.partition = partition
        self.client_sample_nums = None  # number of samples for each client, a list or numpy vector
        self.rng = np.random.default_rng(seed)

        # set balance/unbalance variables
        self.balance = balance
        if balance is True:
            self.unbalance_sgm = 0
        elif balance is False:
            assert isinstance(unbalance_sgm, float), \
                f"'unbalance_sgm' for Log-Normal distribution should be float, " \
                f"not {type(unbalance_sgm)}"
            self.unbalance_sgm = unbalance_sgm
        else:
            raise ValueError(f"'balance' should only be bool, not {type(balance)}.")

        # set iid/non-iid variables
        if self.partition == "iid":
            self.niid_method = None
            self.dirichlet = None
            self.num_shards = None

        elif self.partition == "niid":
            self.niid_method = niid_method
            if self.niid_method == "shards":
                self.dirichlet = None
                assert isinstance(num_shards, int), \
                    f"num_shards should be int for non-iid partition, not {type(num_shards)}."
                self.num_shards = num_shards
            elif self.niid_method == "dirichlet":
                assert isinstance(dirichlet, float), \
                    f"'dirichlet' for non-iid partition using Dirichlet distribution should be " \
                    f"float, not {type(dirichlet)}"
                self.num_shards = None
            else:
                raise ValueError(
                    f"'niid_method' can only be 'shards' or 'dirichlet', {niid_method} is not "
                    f"valid.")

        self._partition()  # perform data partition according to setting
        pass

    def _partition(self):
        if self.balance is True and self.partition == "iid":
            self._iid_balance()
        elif self.balance is False and self.partition == "iid":
            self._iid_unbalance()
        elif self.balance is True and self.partition == "niid":
            self._niid_balance()
        else:  # for balance and niid partition
            self._niid_unbalance()

    def _iid_balance(self):
        rand_perm = self.rng.permutation(self.num_samples)
        num_samples_per_client = int(self.num_samples / self.num_clients)
        self.client_sample_nums = np.ones(self.num_clients) * num_samples_per_client
        num_cumsum = np.concatenate(([0], np.cumsum(self.client_sample_nums))).astype(int)
        self.client_dict = self._slice_indices(num_cumsum, rand_perm)

    def _iid_unbalance(self):
        rand_perm = self.rng.permutation(self.num_samples)
        num_samples_per_client = int(self.num_samples / self.num_clients)
        client_sample_nums = self.rng.lognormal(mean=np.log(num_samples_per_client),
                                                sigma=self.unbalance_sgm,
                                                size=self.num_clients)
        client_sample_nums = (
                client_sample_nums / np.sum(client_sample_nums) * self.num_samples).astype(int)
        diff = np.sum(client_sample_nums) - self.num_samples  # diff <= 0

        # Add/Subtract the excess number starting from first client
        if diff != 0:
            for cid in range(self.num_clients):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break

        self.client_sample_nums = client_sample_nums
        num_cumsum = np.concatenate(([0], np.cumsum(self.client_sample_nums))).astype(int)
        self.client_dict = self._slice_indices(num_cumsum, rand_perm)

    def _niid_balance(self):
        if self.niid_method == "shards":
            size_shard = int(self.num_samples / self.num_shards)
            if self.num_samples % self.num_shards != 0:
                warnings.warn("warning: length of dataset isn't divided exactly by num_shards. "
                              "Some samples will be dropped.")

            shards_per_client = int(self.num_shards / self.num_clients)
            if self.num_shards % self.num_clients != 0:
                warnings.warn("warning: num_shards isn't divided exactly by num_clients. "
                              "Some shards will be dropped.")

            indices = np.arange(self.num_samples)
            # sort sample indices according to labels
            indices_targets = np.vstack((indices, self.targets))
            indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
            # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
            sorted_indices = indices_targets[0, :]

            # permute shards idx, and slice shards_per_client shards for each client
            rand_perm = self.rng.permutation(self.num_shards)
            num_client_shards = np.ones(self.num_clients) * shards_per_client
            # sample index must be int
            num_cumsum = np.concatenate(([0], np.cumsum(num_client_shards))).astype(int)
            # shard indices for each client
            client_shards_dict = self._slice_indices(num_cumsum, rand_perm)

            # map shard idx to sample idx for each client
            client_dict = dict()
            for cid in range(self.num_clients):
                shards_set = client_shards_dict[cid]
                current_indices = [
                    sorted_indices[shard_id * size_shard: (shard_id + 1) * size_shard]
                    for shard_id in shards_set]
                client_dict[cid] = np.concatenate(current_indices, axis=0)

        else:  # Dirichlet for non-iid
            pass

        self.client_dict = client_dict

    def _niid_unbalance(self):
        pass

    def _slice_indices(self, num_cumsum, rand_perm):
        client_dict = dict()
        for cid in range(self.num_clients):
            client_dict[cid] = rand_perm[num_cumsum[cid]: num_cumsum[cid + 1]]
        return client_dict

    def __getitem__(self, index):
        """

        Args:
            index (int): Client ID.

        Returns:
            list: List of sample indices for client ID ``index``.

        """
        return self.client_dict[index]

    def __len__(self):
        return len(self.client_dict)
