import os
import torch
import torchvision.transforms as transforms
import numpy as np

from fedlab_utils.dataset.dataset import BaseDataset
from fedlab_utils.dataset.data_read_util import read_data
from fedlab_utils.dataset.language_utils import word_to_indices, letter_to_index


def get_tarin_test_data(root='../../../data/shakespeare/data/'):
    """Get train data and test data for shakespeare from ``root`` path, preprocess data format as model need.

    Notes:
        Please check data downloaded and preprocessed completely before running this method.

        The shell script for data is in ``FedLab/data/shakespeare/run.sh``,
        preprocessed data should be in `root`/train and `root`/test.

    Args:
        root (str, optional): path string contains `train` and `test` folders for train data and test data.
            Defaluts to '../../../data/shakespeare/data/', is equals to 'FedLab/data/shakespeare/data/'

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
    """for all word strings in raw_x, process each char to index in ALL_LETTERS

    Args:
        raw_x (list[string]): contains a list of word strings to process

    Returns:
        x (list[list]): int indices list for words in raw_x in ALL_LETTERS
    """
    x = [word_to_indices(word) for word in raw_x]
    return x


def process_y(raw_y):
    """for all labels(next char) in raw_y, process them to index in ALL_LETTERS

    Args:
        raw_y (list[char]): contains a list of label to process

    Returns:
        y (list[int]): the index for all chars in raw_y in ALL_LETTERS list

    """
    y = [letter_to_index(c) for c in raw_y]
    return y


def get_dataloader_shakespeare(client_id=0, batch_size=128):
    """Get shakespeare dataloader with ``batch_size`` param for client with ``client_id``

    Args:
        client_id (int, optional): assigned client_id to get dataloader for this client. Defaults to 0
        batch_size (int, optional): the number of batch size for dataloader. Defaluts to 128

    Returns:
        A tuple with train dataloader and test dataloader for the client with `client_id`::

        (
            'trainloader' (torch.utils.data.DataLoader): dataloader for train dataset to client with client_id,
            'testloader' (torch.utils.data.DataLoader): dataloader for test dataset to client with client_id
        )
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

if __name__ == '__main__':
    trainloader, testloader = get_dataloader_shakespeare()