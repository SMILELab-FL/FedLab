"""Get

"""
import torch
import os
import numpy as np
import torchvision.transforms as transforms
from fedlab_utils.dataset.dataset import BaseDataset
from fedlab_utils.dataset.data_read_util import read_data


def get_tarin_test_data(root='../../../data/femnist/data'):
    """Get train data and test data for femnist from ``root`` path, preprocess data format as model need.

    Notes:
        Please check data downloaded and preprocessed completely before running this method.

        The shell script for data is in ``FedLab/data/femnist/run.sh``,
        preprocessed data should be in `root`/train and `root`/test.

    Args:
        root (str, optional): path string contains `train` and `test` folders for train data and test data.
            Defaluts to '../../../data/femnist/data/', is equals to 'FedLab/data/femnist/data/'

    Returns:
        A tuple contains the client number, train data and test data for each clients::

        (
            'client_num' (int): the client number of split data by LEAF,
            'train_data_x_dict' (dict[int, Tensor]): A train input dict mapping keys to the corresponding client id,
            'train_data_y_dict' (dict[int, Tensor]): A train label dict mapping keys to the corresponding client id,
            'test_data_x_dict' (dict[int, Tensor]): A test input dict mapping keys to the corresponding client id,
            'test_data_y_dict' (dict[int, Tensor]): A test label dict mapping keys to the corresponding client id.
        )
    Raises:
        FileNotFoundError: [Errno 2] No such file or directory: '`root`/train' or '`root`/test'
    """
    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')

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


def get_dataloader_femnist(client_id=0, batch_size=128):
    """Get femnist dataloader with ``batch_size`` param for client with ``client_id``

    Args:
        client_id (int, optional): assigned client_id to get dataloader for this client. Defaults to 0
        batch_size (int, optional): the number of batch size for dataloader. Defaults to 128

    Returns:
        A tuple with train dataloader and test dataloader for the client with `client_id`::

        (
            'trainloader' (torch.utils.data.DataLoader): dataloader for train dataset to client with client_id,
            'testloader' (torch.utils.data.DataLoader): dataloader for test dataset to client with client_id
        )
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
