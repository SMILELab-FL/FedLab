# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import json
import pynvml
import numpy as np
import pickle
from collections import Counter


class AverageMeter(object):
    """Record metrics information"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def evaluate(model, criterion, test_loader):
    """Evaluate classify task model accuracy."""
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

    return loss_.sum, acc_.avg


def read_config_from_json(json_file: str, user_name: str):
    """Read config from `json_file` to get config for `user_name`

    Args:
        json_file (str): path for json_file
        user_name (str): read config for this user, it can be 'server' or 'client_id'

    Returns:
        a tuple with ip, port, world_size, rank about user with `user_name`

    Examples:
        read_config_from_json('../../../tests/data/config.json', 'server')

    Notes:
        config.json example as follows
        {
          "server": {
            "ip" : "127.0.0.1",
            "port": "3002",
            "world_size": 3,
            "rank": 0
          },
          "client_0": {
            "ip": "127.0.0.1",
            "port": "3002",
            "world_size": 3,
            "rank": 1
          },
          "client_1": {
            "ip": "127.0.0.1",
            "port": "3002",
            "world_size": 3,
            "rank": 2
          }
        }
    """
    with open(json_file) as f:
        config = json.load(f)
    config_info = config[user_name]
    return (config_info["ip"], config_info["port"], config_info["world_size"],
            config_info["rank"])


def get_best_gpu():
    """Return gpu (:class:`torch.device`) with largest free memory."""
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    deviceMemory = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    return torch.device("cuda:%d" % (best_device_index))


def save_dict(dict, path):
    with open(path, 'wb') as f:
        pickle.dump(dict, f)


def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def partition_report(targets, data_indices, class_num=None, verbose=True, file=None):
    """Generate data partition report for clients in ``data_indices``.

    Generate data partition report for each client according to ``data_indices``, including
    ratio of each class and dataset size in current client. Report can be printed in screen or into
    file. The output format is comma-separated values which can be read by :func:`pandas.read_csv`
    or :func:`csv.reader`.

    Args:
        targets (list or numpy.ndarray): Targets for all data samples, with each element is in range of ``0`` to ``class_num-1``.
        data_indices (dict): Dict of ``client_id: [data indices]``.
        class_num (int, optional): Total number of classes. If set to ``None``, then ``class_num = max(targets) + 1``.
        verbose (bool, optional): Whether print data partition report in screen. Default as ``True``.
        file (str, optional): Output file name of data partition report. If ``None``, then no output in file. Default as ``None``.

    Examples:
        First generate synthetic data labels and data partition to obtain ``data_indices``
        (``{ client_id: sample indices}``):

        >>> sample_num = 15
        >>> class_num = 4
        >>> clients_num = 3
        >>> num_per_client = int(sample_num/clients_num)
        >>> labels = np.random.randint(class_num, size=sample_num)  # generate 15 labels, each label is 0 to 3
        >>> rand_per = np.random.permutation(sample_num)
        >>> # partition synthetic data into 3 clients
        >>> data_indices = {0: rand_per[0:num_per_client],
        ...                 1: rand_per[num_per_client:num_per_client*2],
        ...                 2: rand_per[num_per_client*2:num_per_client*3]}

        Check ``data_indices`` may look like:

        >>> data_indices
        {0: array([8, 6, 5, 7, 2]),
         1: array([ 3, 10, 14,  4,  1]),
         2: array([13,  9, 12, 11,  0])}

        Now generate partition report for each client and each class:

        >>> partition_report(labels, data_indices, class_num=class_num, verbose=True, file=None)
        Class frequencies:
        client,class0,class1,class2,class3,Amount
        Client   0,0.200,0.00,0.200,0.600,5
        Client   1,0.400,0.200,0.200,0.200,5
        Client   2,0.00,0.400,0.400,0.200,5

    """
    if not verbose and file is None:
        print("No partition report generated")
        return

    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    if not class_num:
        class_num = max(targets) + 1

    sorted_cid = sorted(data_indices.keys())  # sort client id in ascending order

    header_line = "Class frequencies:"
    col_name = "client," + ','.join([f"class{i}" for i in range(class_num)]) + ",Amount"

    if verbose:
        print(header_line)
        print(col_name)
    if file is not None:
        reports = [header_line, col_name]
    else:
        reports = None

    for client_id in sorted_cid:
        indices = data_indices[client_id]
        client_targets = targets[indices]
        client_sample_num = len(indices)  # total number of samples of current client
        client_target_cnt = Counter(client_targets)  # { cls1: num1, cls2: num2, ... }

        report_line = f"Client {client_id:3d}," + \
                      ','.join([
                          f"{client_target_cnt[cls] / client_sample_num:.3f}" if cls in client_target_cnt else "0.00"
                          for cls in range(class_num)]) + \
                      f",{client_sample_num}"
        if verbose:
            print(report_line)
        if file is not None:
            reports.append(report_line)

    if file is not None:
        fh = open(file, "w")
        fh.write("\n".join(reports))
        fh.close()
