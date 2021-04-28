"""
functions associated with data and dataset operations

暂时放着，未修改
"""
import warnings

import numpy as np
from torchvision import datasets, transforms
from copy import deepcopy


def dataset_noniid(dataset, num_clients, num_shards):
    """
    将dataset划分为每块大小为num_shards的非独立同分布的块
    按块数量平均分配给num_clients数量的参与者

    返回 各参与者数据集在dataset对应的索引表
    """
    size_of_shards = int(len(dataset)/num_shards)
    if len(dataset) % num_shards != 0:
        warnings.warn(
            "warning: the length of dataset isn't divided exactly by num_shard.some samples will be wasted.")
    # the number of shards that each one of clients can get
    shard_pc = int(num_shards/num_clients)
    if num_shards % num_clients != 0:
        warnings.warn(
            "warning: num_shard isn't divided exacly by num_clients. some samples will be wasted.")

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
                (dict_users[i], idxs[rand*size_of_shards:(rand+1)*size_of_shards]), axis=0)
    
    return dict_users



def dataset_random(dataset, num_clients):
    """
    将dataset随机划分分配给num_clients数量的参与者
    返回 各参与者数据集在dataset对应的索引表
    """
    num_items = int(len(dataset)/num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(
            all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def get_dataset(name):
    if name == "cifar10":
        dataset_train = datasets.CIFAR10(
            '../datasets/cifar10/', train=True)
        dataset_test = datasets.CIFAR10(
            '../datasets/cifar10/', train=False)
        return dataset_train, dataset_test
    elif name == "mnist":
        dataset_train = datasets.MNIST('../data/mnist/', train=True)
        dataset_test = datasets.MNIST('../data/mnist/', train=False)
        return dataset_train, dataset_test

    else:
        raise ValueError("Invalid dataset name")


def divide_dataset(dict_users, dataset_train):
    """
    返回数据集分割后的元数据组
    """
    datasets = []
    data = dataset_train.data
    label = np.array(dataset_train.targets)
    for _, dic in dict_users.items():
        dic = np.array(list(dic))
        client_data = data[dic]
        client_label = list(label[dic])
        client_dataset = (client_data, client_label)
        datasets.append(client_dataset)
    return datasets


def cut_dataset(dataset, cutsize, transform=None):
    """ cut a base dataset into 2 pieces """
    data = dataset.data
    targets = dataset.targets

    cut_data = deepcopy(data[0:cutsize])
    remain_data = deepcopy(data[cutsize:])

    cut_targets = deepcopy(targets[0:cutsize])
    remain_targets = deepcopy(targets[cutsize:])

    cut_dataset = BaseDataset(
        data=cut_data, targets=cut_targets, transform=transform)
    remain_dataset = BaseDataset(
        data=remain_data, targets=remain_targets, transform=transform)

    return cut_dataset, remain_dataset


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))
