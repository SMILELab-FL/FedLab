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
"""functions associated with data and dataset slicing"""

import warnings
import numpy as np


def noniid_slicing(dataset, num_clients, num_shards):
    """Slice a dataset 
    
    Args:
        dataset (torch.utils.data.Dataset): a dataset for slicing
        num_clients (int):  the number of client.
        num_shards (int): the number of shards. 
    
    Notes:
        The size of a shard = int(len(dataset)/num_shards).
        Each client will get int(num_shards/num_clients) shards.

    Returns：
        dict: { '0': indices of dataset,
                '1': indices of dataset,
                ...
                'k': indices of dataset }
    """
    size_of_shards = int(len(dataset) / num_shards)
    if len(dataset) % num_shards != 0:
        warnings.warn(
            "warning: the length of dataset isn't divided exactly by num_shard.some samples will be wasted."
        )
    # the number of shards that each one of clients can get
    shard_pc = int(num_shards / num_clients)
    if num_shards % num_clients != 0:
        warnings.warn(
            "warning: num_shard isn't divided exacly by num_clients. some samples will be wasted."
        )

    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}

    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 将标签按索引排序，调换顺序
    idxs = idxs_labels[0, :]

    # assign
    idx_shard = [i for i in range(num_shards)]
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, shard_pc, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i],
                 idxs[rand * size_of_shards:(rand + 1) * size_of_shards]),
                axis=0)

    return dict_users


def random_slicing(dataset, num_clients):
    """Slice a dataset randomly and equally

    Args：
        dataset (torch.utils.data.Dataset): a dataset for slicing.
        num_clients (int):  the number of client.

    Returns：
        dict: { '0': indices of dataset,
                '1': indices of dataset,
                ...
                'k': indices of dataset }
    """
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = list(
            np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users

def divide_dataset(dataset, slicing_dict, to_file=None):
    """cut a dataset
    
    Args:
        dataset (torch.utils.data.Dataset): a dataset for slicing.
        slicing_dict (dict): {id:indices}
    
    Returns:
        [(data_0, label_0), (data_1, label_1), ... , (data_k, label_k)]
    """
    datasets = []
    data = dataset.data
    label = np.array(dataset.targets)
    for _, dic in slicing_dict.items():
        dic = np.array(list(dic))
        client_data = data[dic]
        client_label = list(label[dic])
        client_dataset = (client_data, client_label)
        datasets.append(client_dataset)
    return datasets
