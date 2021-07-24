"""
    get data and dataloader for dataset in LEAF
"""

import os
import torch
from fedlab_utils.dataset.dataset import BaseDataset
from fedlab_benchmarks.datasets.leaf_data_process.data_read_util import read_data

from fedlab_benchmarks.datasets.leaf_data_process.femnist import process as process_femnist
from fedlab_benchmarks.datasets.leaf_data_process.shakespeare import process as process_shakespeare


def get_train_test_data(root, client_id=0, process_x=None, process_y=None):
    """Get train data and test data for femnist from ``root`` path, preprocess data by given processed method.

    Notes:
        Please check data downloaded and preprocessed completely before running this method.

        The shell script for data is in ``FedLab/fedlab_benchmarks/datasets/leaf_data/femnist/run.sh``,
        preprocessed data should be in `root`/train and `root`/test.

    Args:
        root (str): path string contains `train` and `test` folders for train data and test data.
        client_id (int): data for client_id. Defaults to 0, the first client
        process_x (optional[callable]): method to process x. Defaults to None
        process_y (optional[callable]): method to process y. Defaults to None

    Returns:
        A tuple contains the client number, train data and test data for each clients

    Examples:
        train_data_x, train_data_y, test_data_x, test_data_y = get_train_test_data('../leaf_data/femnist/data')

    Raises:
        FileNotFoundError: [Errno 2] No such file or directory: '`root`/train' or '`root`/test'
    """
    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')

    users, groups, train_data, test_data = read_data(train_path, test_path)

    client_name = users[client_id]
    if process_x is not None:
        train_data_x = process_x(train_data[client_name]['x'])
        test_data_x = process_x(test_data[client_name]['x'])
    else:
        train_data_x = train_data[client_name]['x']
        test_data_x = test_data[client_name]['x']
    if process_y is not None:
        train_data_y = process_y(train_data[client_name]['y'])
        test_data_y = process_y(test_data[client_name]['y'])
    else:
        train_data_y = train_data[client_name]['y']
        test_data_y = test_data[client_name]['y']

    return train_data_x, train_data_y, test_data_x, test_data_y


def get_dataloader(dataset, client_id=0, batch_size=128):
    """Get shakespeare dataloader with ``batch_size`` param for client with ``client_id``

    Args:
        dataset (str): dataloader for dataset
        client_id (int, optional): assigned client_id to get dataloader for this client. Defaults to 0
        batch_size (int, optional): the number of batch size for dataloader. Defaults to 128

    Returns:
        A tuple with train dataloader and test dataloader for the client with `client_id`
    """

    if dataset == 'femnist':
        train_data_x, train_data_y, test_data_x, test_data_y = get_train_test_data('../leaf_data/femnist/data',
                                                                                   client_id,
                                                                                   process_femnist.process_x,
                                                                                   process_femnist.process_y)
    elif dataset == 'shakespeare':
        train_data_x, train_data_y, test_data_x, test_data_y = get_train_test_data('../leaf_data/shakespeare/data',
                                                                                   client_id,
                                                                                   process_shakespeare.process_x,
                                                                                   process_shakespeare.process_y)

    trainset = BaseDataset(data=train_data_x, targets=train_data_y)
    testset = BaseDataset(data=test_data_x, targets=test_data_y)

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
