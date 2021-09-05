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


class AverageMeter(object):
    """Record train infomation"""
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