import os
import torch
import torchvision.transforms as transforms
import numpy as np

from fedlab_utils.dataset.dataset import BaseDataset
from fedlab_utils.dataset.data_read_util import read_data
from fedlab_utils.dataset.language_utils import word_to_indices, letter_to_index


def get_tarin_test_data(root='../../../data/shakespeare/data/'):
    """
    get train data and test data for Shakespeare,
    before it, we should run script to get partioned data, the path is  /data/shakespeare/download.sh
    Args:
        root: path contains train and test data folders

    Returns:

    """
    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')

    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]

    # client's data map
    train_data_x_dict = dict()
    train_data_y_dict = dict()
    test_data_x_dict = dict()
    test_data_y_dict = dict()

    client_idx = 0
    for u, g in zip(users, groups):
        train_data_x_dict[client_idx] = torch.from_numpy(np.asarray(process_x(train_data[u]['x'])))
        train_data_y_dict[client_idx] = torch.from_numpy(np.asarray(process_y(train_data[u]['y'])))
        test_data_x_dict[client_idx] = torch.from_numpy(np.asarray(process_x(test_data[u]['x'])))
        test_data_y_dict[client_idx] = torch.from_numpy(np.asarray(process_y(test_data[u]['y'])))

        client_idx += 1
    client_num = client_idx
    return client_num, train_data_x_dict, train_data_y_dict, test_data_x_dict, test_data_y_dict


def process_x(raw_x):
    x = [word_to_indices(word) for word in raw_x]
    # x_batch = np.array(x_batch)
    return x


def process_y(raw_y):
    y = [letter_to_index(c) for c in raw_y]
    return y


def get_dataloader_shakespeare(client_id=None, batch_size=128):
    """
    get shakespeare dataloader for an assigned client or all data
    Args:
        client_id: get dataloader for assigned client
        batch_size:

    Returns:
        dataloader for one client with client_id
        or return dataloader with all data together

    """

    client_num, train_data_x_dict, train_data_y_dict, test_data_x_dict, test_data_y_dict= get_tarin_test_data()

    if client_id >= client_num:
        print("uncorrect client id, larger than total client number")
        return None
    trainset = BaseDataset(data=train_data_x_dict[client_id], targets=train_data_y_dict[client_id])
    testset = BaseDataset(data=test_data_x_dict[client_id], targets=test_data_y_dict[client_id])

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        drop_last=True)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=len(testset),
        drop_last=False,
        shuffle=False)
    return trainloader, testloader
