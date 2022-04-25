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

import numpy as np
import pandas as pd
import warnings


def split_indices(num_cumsum, rand_perm):
    """Splice the sample index list given number of each client.

    Args:
        num_cumsum (np.ndarray): Cumulative sum of sample number for each client.
        rand_perm (list): List of random sample index.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    client_indices_pairs = [(cid, idxs) for cid, idxs in
                            enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    client_dict = dict(client_indices_pairs)
    return client_dict


def balance_split(num_clients, num_samples):
    """Assign same sample sample for each client.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    num_samples_per_client = int(num_samples / num_clients)
    client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(
        int)
    return client_sample_nums


def lognormal_unbalance_split(num_clients, num_samples, unbalance_sgm):
    """Assign different sample number for each client using Log-Normal distribution.

    Sample numbers for clients are drawn from Log-Normal distribution.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.
        unbalance_sgm (float): Log-normal variance. When equals to ``0``, the partition is equal to :func:`balance_partition`.

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    num_samples_per_client = int(num_samples / num_clients)
    if unbalance_sgm != 0:
        client_sample_nums = np.random.lognormal(mean=np.log(num_samples_per_client),
                                                 sigma=unbalance_sgm,
                                                 size=num_clients)
        client_sample_nums = (
                client_sample_nums / np.sum(client_sample_nums) * num_samples).astype(int)
        diff = np.sum(client_sample_nums) - num_samples  # diff <= 0

        # Add/Subtract the excess number starting from first client
        if diff != 0:
            for cid in range(num_clients):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break
    else:
        client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)

    return client_sample_nums


def dirichlet_unbalance_split(num_clients, num_samples, alpha):
    """Assign different sample number for each client using Log-Normal distribution.

    Sample numbers for clients are drawn from Log-Normal distribution.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.
        alpha (float): Dirichlet concentration parameter

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    min_size = 0
    while min_size < 10:
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        min_size = np.min(proportions * num_samples)

    client_sample_nums = (proportions * num_samples).astype(int)
    return client_sample_nums


def homo_partition(client_sample_nums, num_samples):
    """Partition data indices in IID way given sample numbers for each clients.

    Args:
        client_sample_nums (numpy.ndarray): Sample numbers for each clients.
        num_samples (int): Number of samples.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    rand_perm = np.random.permutation(num_samples)
    num_cumsum = np.cumsum(client_sample_nums).astype(int)
    client_dict = split_indices(num_cumsum, rand_perm)
    return client_dict


def hetero_dir_partition(targets, num_clients, num_classes, dir_alpha, min_require_size=None):
    """

    Non-iid partition based on Dirichlet distribution. The method is from "hetero-dir" partition of
    `Bayesian Nonparametric Federated Learning of Neural Networks <https://arxiv.org/abs/1905.12022>`_
    and `Federated Learning with Matched Averaging <https://arxiv.org/abs/2002.06440>`_.

    This method simulates heterogeneous partition for which number of data points and class
    proportions are unbalanced. Samples will be partitioned into :math:`J` clients by sampling
    :math:`p_k \sim \\text{Dir}_{J}({\\alpha})` and allocating a :math:`p_{p,j}` proportion of the
    samples of class :math:`k` to local client :math:`j`.

    Sample number for each client is decided in this function.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    if min_require_size is None:
        min_require_size = num_classes

    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_samples = targets.shape[0]

    min_size = 0
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        # for each class in the dataset
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(
                np.repeat(dir_alpha, num_clients))
            # Balance
            proportions = np.array(
                [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in
                 zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_dict = dict()
    for cid in range(num_clients):
        np.random.shuffle(idx_batch[cid])
        client_dict[cid] = np.array(idx_batch[cid])

    return client_dict


def shards_partition(targets, num_clients, num_shards):
    """Non-iid partition used in FedAvg `paper <https://arxiv.org/abs/1602.05629>`_.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        num_shards (int): Number of shards in partition.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_samples = targets.shape[0]

    size_shard = int(num_samples / num_shards)
    if num_samples % num_shards != 0:
        warnings.warn("warning: length of dataset isn't divided exactly by num_shards. "
                      "Some samples will be dropped.")

    shards_per_client = int(num_shards / num_clients)
    if num_shards % num_clients != 0:
        warnings.warn("warning: num_shards isn't divided exactly by num_clients. "
                      "Some shards will be dropped.")

    indices = np.arange(num_samples)
    # sort sample indices according to labels
    indices_targets = np.vstack((indices, targets))
    indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
    # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
    sorted_indices = indices_targets[0, :]

    # permute shards idx, and slice shards_per_client shards for each client
    rand_perm = np.random.permutation(num_shards)
    num_client_shards = np.ones(num_clients) * shards_per_client
    # sample index must be int
    num_cumsum = np.cumsum(num_client_shards).astype(int)
    # shard indices for each client
    client_shards_dict = split_indices(num_cumsum, rand_perm)

    # map shard idx to sample idx for each client
    client_dict = dict()
    for cid in range(num_clients):
        shards_set = client_shards_dict[cid]
        current_indices = [
            sorted_indices[shard_id * size_shard: (shard_id + 1) * size_shard]
            for shard_id in shards_set]
        client_dict[cid] = np.concatenate(current_indices, axis=0)

    return client_dict


def client_inner_dirichlet_partition(targets, num_clients, num_classes, dir_alpha,
                                     client_sample_nums, verbose=True):
    """Non-iid Dirichlet partition.

    The method is from The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.
    This function can be used by given specific sample number for all clients ``client_sample_nums``.
    It's different from :func:`hetero_dir_partition`.

    Args:
        targets (list or numpy.ndarray): Sample targets.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        client_sample_nums (numpy.ndarray): A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    rand_perm = np.random.permutation(targets.shape[0])
    targets = targets[rand_perm]

    class_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes,
                                       size=num_clients)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                      range(num_clients)]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print('Remaining Data: %d' % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict


def label_skew_quantity_based_partition(targets, num_clients, num_classes, major_classes_num):
    """Label-skew:quantity-based partition.

    For details, please check `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_.

    Args:
        targets (np.ndarray): Labels od dataset.
        num_clients (int): Number of clients.
        num_classes (int): Number of unique classes.
        major_classes_num (int): Number of classes for each client, should be less then ``num_classes``.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    idx_batch = [np.ndarray(0, dtype=np.int64) for _ in range(num_clients)]
    # only for major_classes_num < num_classes.
    # if major_classes_num = num_classes, it equals to IID partition
    times = [0 for _ in range(num_classes)]
    contain = []
    for cid in range(num_clients):
        current = [cid % num_classes]
        times[cid % num_classes] += 1
        j = 1
        while j < major_classes_num:
            ind = np.random.randint(num_classes)
            if ind not in current:
                j += 1
                current.append(ind)
                times[ind] += 1
        contain.append(current)

    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        split = np.array_split(idx_k, times[k])
        ids = 0
        for cid in range(num_clients):
            if k in contain[cid]:
                idx_batch[cid] = np.append(idx_batch[cid], split[ids])
                ids += 1

    client_dict = {cid: idx_batch[cid] for cid in range(num_clients)}
    return client_dict


def fcube_synthetic_partition(data):
    """Feature-distribution-skew:synthetic partition.

    Synthetic partition for FCUBE dataset. This partition is from `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_.

    Args:
        data (np.ndarray): Data of dataset :class:`FCUBE`.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    num_clients = 4
    client_indices = [[] for _ in range(num_clients)]
    for idx, sample in enumerate(data):
        p1, p2, p3 = sample
        if (p1 > 0 and p2 > 0 and p3 > 0) or (p1 < 0 and p2 < 0 and p3 < 0):
            client_indices[0].append(idx)
        elif (p1 > 0 and p2 > 0 and p3 < 0) or (p1 < 0 and p2 < 0 and p3 > 0):
            client_indices[1].append(idx)
        elif (p1 > 0 and p2 < 0 and p3 > 0) or (p1 < 0 and p2 > 0 and p3 < 0):
            client_indices[2].append(idx)
        else:
            client_indices[3].append(idx)
    client_dict = {cid: np.array(client_indices[cid]).astype(int) for cid in range(num_clients)}
    return client_dict


def samples_num_count(client_dict, num_clients):
    """Return sample count for all clients in ``client_dict``.

    Args:
        client_dict (dict): Data partition result for different clients.
        num_clients (int): Total number of clients.

    Returns:
        pandas.DataFrame

    """
    client_samples_nums = [[cid, client_dict[cid].shape[0]] for cid in
                           range(num_clients)]
    client_sample_count = pd.DataFrame(data=client_samples_nums,
                                       columns=['client', 'num_samples']).set_index('client')
    return client_sample_count
