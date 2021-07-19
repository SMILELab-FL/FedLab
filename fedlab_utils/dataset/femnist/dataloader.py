import torch
import os
import numpy as np
import torchvision.transforms as transforms
from fedlab_utils.dataset.dataset import BaseDataset
from fedlab_utils.dataset.data_read_util import read_data


def get_tarin_test_data(root='../../../data/femnist/data/'):
    """
    get train data and test data for femnist,
    before it, we should run script to get partioned data, the path is  /data/femnist/download.sh
    Args:
        root: path contains train and test data folders

    Returns:

    """
    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')
    if not os.path.exists(train_path):
        print("please check if data have been downloaded correctly, "
              "you can got to fedlab_utils/data to re-download by run preprocess.sh with some params")

    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_x_dict = dict()
    train_data_y_dict = dict()
    test_data_x_dict = dict()
    test_data_y_dict = dict()
    client_idx = 0
    for u, g in zip(users, groups):
        # get each client's data, and transform to the format of mnist data
        train_data_x_dict[client_idx] = torch.reshape(torch.from_numpy(np.asarray(train_data[u]['x'])), (-1, 1, 28, 28))
        train_data_x_dict[client_idx] = train_data_x_dict[client_idx].to(torch.float32)
        train_data_y_dict[client_idx] = torch.from_numpy(np.asarray(train_data[u]['y'])).to(torch.long)
        test_data_x_dict[client_idx] = torch.reshape(torch.from_numpy(np.asarray(test_data[u]['x'])), (-1, 1, 28, 28))
        test_data_x_dict[client_idx] = test_data_x_dict[client_idx].to(torch.float32)
        test_data_y_dict[client_idx] = torch.from_numpy(np.asarray(test_data[u]['y'])).to(torch.long)

        client_idx += 1
    client_num = client_idx
    return client_num, train_data_x_dict, train_data_y_dict, test_data_x_dict, test_data_y_dict


def get_dataloader_femnist(client_id=None, batch_size=128):
    """
    get femnist dataloader for an assigned client or all data
    Args:
        client_id: get dataloader for assigned client
        batch_size:

    Returns:
        dataloader for one client with client_id
        or return dataloader with all data together

    """

    client_num, train_data_x_dict, train_data_y_dict, test_data_x_dict, test_data_y_dict = get_tarin_test_data()

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
