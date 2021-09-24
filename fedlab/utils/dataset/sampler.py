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

from copy import deepcopy
import random
import torch
import torch.distributed as dist
from ..functional import load_dict


class SubsetSampler(torch.utils.data.Sampler):
    """Samples elements from a given list of indices, always in the same order once initialized.

    It is a :class:`Sampler` used in :class:`Dataloader`, that each partition will be fixed once initialized.

    Args:
        indices (list[int]): Indices in the whole set selected for subset
        shuffle (bool): shuffle the indices or not.
    """

    def __init__(self, indices, shuffle=False) -> None:
        self.indices = indices
        if shuffle is True:
            random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class RawPartitionSampler(torch.utils.data.Sampler):
    """Partition dataset according to ``num_replicas``.
    
    Every client get a equal shard of dataset.

    Args:
        dataset (torch.utils.data.Dataset):
        client_id (int):
        num_replicas (int, optional): Number of data replications. Default ``None`` means total number of client processes.
    """

    def __init__(self, dataset, client_id, num_replicas=None):

        self.indices = [index for index in range(len(dataset))]

        if num_replicas is None:
            self.num_replicas = dist.get_world_size() - 1  # world size includes 1 server process
        else:
            self.num_replicas = num_replicas

        assert isinstance(client_id, int), "client_id should be int, not {}".format(type(client_id))
        assert client_id <= self.num_replicas, "client_id should not be greater than num_replicas={}".format(
            self.num_replicas)
        self.id = client_id

        self.num_samples = int(len(dataset) / self.num_replicas)

    def __iter__(self):

        local_indices = self.indices[(self.id - 1) * self.num_samples:self.id *
                                                                      self.num_samples]
        assert len(local_indices) == self.num_samples
        return iter(local_indices)

    def __len__(self):
        return self.num_samples


class DictFileSampler(torch.utils.data.Sampler):
    """Get data sample indices given client id from data file with dict."""

    def __init__(self, dict_file, client_id):
        data_indices = load_dict(dict_file)
        self.indices = data_indices[client_id]

    def __iter__(self):
        return iter(self.indices)  # TODO: why no shuffle here?

    def __len__(self):
        return len(self.indices)
