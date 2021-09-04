"""
    This is modified by [LEAF/models/utils/model_utils.py]
    https://github.com/TalwalkarLab/leaf/blob/master/models/utils/model_utils.py

    Read data for `../leaf_data` directory processed json files.
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

    Examples:
        read_dir(data_dir="../data/femnist/data/train"):
    """
    # Splicing absolute path
    data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), data_dir)

    groups = []
    client_name2data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        client_name2data.update(cdata['user_data'])

    # generate clients_id_str - client_id_index map
    clients_name = list(sorted(client_name2data.keys()))
    clients_id = list(range(len(clients_name)))
    client_id2name = dict(zip(clients_id, clients_name))

    return client_id2name, groups, client_name2data
