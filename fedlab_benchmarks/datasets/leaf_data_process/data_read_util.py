"""
    This is modified by [LEAF/models/utils/model_utils.py](https://github.com/TalwalkarLab/leaf/blob/master/models/utils/model_utils.py)
    Read data for leaf_data_process processed json files.
    Methods:
        read_dir(data_dir): Read .json file from ``data_fir``
        read_data(train_data_dir, test_data_dir): parses data in given train and test data directories
"""
import json
import os
from collections import defaultdict


def read_dir(data_dir):
    """ Read .json file from ``data_fir``

    Args:
        data_dir: directory contains json files

    Returns:
        clients name dict mapping keys to id, groups list for each clients, a dict data mapping keys to client
    """
    # Splicing absolute path
    data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), data_dir)
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    # generate clients_id_str - client_id_index map
    clients_name = list(sorted(data.keys()))
    clients_id = list(range(len(clients_name)))
    clients = dict(zip(clients_id, clients_name))

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
        client id list, group id list, dictionary of train data, dictionary of test data
    """

    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data
