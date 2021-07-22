import os
import torch
from fedlab_utils.dataset.dataset import BaseDataset
from fedlab_utils.dataset.leaf.data_read_util import read_data

from fedlab_utils.dataset.leaf.process.femnist import process as process_femnist
from fedlab_utils.dataset.leaf.process.shakespeare import process as process_shakespeare
from fedlab_utils.dataset.leaf.process.sent140 import process as process_sent140


def get_tarin_test_data(root, process_x, process_y):
    """Get train data and test data for femnist from ``root`` path, preprocess data by given processed method.

    Notes:
        Please check data downloaded and preprocessed completely before running this method.

        The shell script for data is in ``FedLab/fedlab_benchmarks/datasets/femnist/run.sh``,
        preprocessed data should be in `root`/train and `root`/test.

    Args:
        processY:

        root (str): path string contains `train` and `test` folders for train data and test data.
        processX ()

    Returns:
        A tuple contains the client number, train data and test data for each clients

    Examples:
        client_num, train_data_x_dict, train_data_y_dict, test_data_x_dict, test_data_y_dict
            = get_tarin_test_data('../../../fedlab_benchmarks/datasets/femnist/data')

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
        # get each client's data, use given process method to transform format
        train_data_x_dict[client_idx] = process_x(train_data[u]['x'])
        train_data_y_dict[client_idx] = process_y(train_data[u]['y'])
        test_data_x_dict[client_idx] = process_x(test_data[u]['x'])
        test_data_y_dict[client_idx] = process_y(test_data[u]['y'])

        client_idx += 1
    client_num = client_idx
    return client_num, train_data_x_dict, train_data_y_dict, test_data_x_dict, test_data_y_dict


def get_dataloader(dataset, client_id=0, batch_size=128):
    """Get shakespeare dataloader with ``batch_size`` param for client with ``client_id``

    Args:
        data_root (str):
        client_id (int, optional): assigned client_id to get dataloader for this client. Defaults to 0
        batch_size (int, optional): the number of batch size for dataloader. Defaluts to 128

    Returns:
        A tuple with train dataloader and test dataloader for the client with `client_id`
    """
    if dataset == 'femnist':
        client_num, train_data_x_dict, train_data_y_dict, test_data_x_dict, test_data_y_dict \
            = get_tarin_test_data('../../../fedlab_benchmarks/datasets/femnist/data',
                                  process_femnist.process_x, process_femnist.process_y)
    elif dataset == 'shakespeare':
        client_num, train_data_x_dict, train_data_y_dict, test_data_x_dict, test_data_y_dict \
            = get_tarin_test_data('../../../fedlab_benchmarks/datasets/shakespeare/data',
                                  process_shakespeare.process_x, process_shakespeare.process_y)
    elif dataset == 'sent140':
        client_num, train_data_x_dict, train_data_y_dict, test_data_x_dict, test_data_y_dict \
            = get_tarin_test_data('../../../fedlab_benchmarks/datasets/sent140/data',
                                  process_sent140.process_x, process_sent140.process_y)

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
    get_dataloader(dataset='shakespeare')