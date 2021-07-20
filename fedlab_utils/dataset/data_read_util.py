""" Read data for leaf processed json files.

"""
import json
import numpy as np
import os
from collections import defaultdict


def batch_data(data, batch_size, seed=100):
    """Generate batch data for leaf data after reading data method

    Args:
        data (dict): client's data, a dict:={'x': [numpy array], 'y': [numpy array]} (on one client)
        batch_size: size of batch
        seed: control random reproduce

    Returns:
        batch_data: contains batch_x and batch_y, which are both numpy array of length: batch_size

    """
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        # batched_x = torch.from_numpy(np.asarray(process_x(batched_x)))
        # batched_y = torch.from_numpy(np.asarray(process_y(batched_y)))
        batch_data.append((batched_x, batched_y))
    return batch_data


def read_dir(data_dir):
    """ Read .json file from ``data_fir``

    Args:
        data_dir: directory contains json files

    Returns:
        clients (list): all clients' id or name list
        groups (list): groups list for each clients, it can be none
        data (dict): a dict data mapping keys to client
    """
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories
        Assumes:

        - the data in the input directories are .json files with
            keys 'users' and 'user_data'
        - the set of train set users is the same as the set of test set users

    Args:
        train_data_dir (str): path string for train data folder
        test_data_dir (str): path string for test data folder

    Returns:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """

    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data
