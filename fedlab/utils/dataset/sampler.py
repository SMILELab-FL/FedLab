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
    """Subset of a dataset at specified indices. 
        Similar to torch.utils.data.dataset.Subset, but this is a Sampler used in Dataloader

    Args:
        indices (list): Indices in the whole set selected for subset
        shuffle (bool): shuffle the indices or not.
    """
    def __init__(self, indices: list, shuffle=False) -> None:
        self.indices = indices
        if shuffle is True:
            random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class RawPartitionSampler(torch.utils.data.Sampler):
    """Partition dataset according to num_replicas.
    
        Every client get a equal shard of dataset.
    """
    def __init__(self, dataset, num_replicas=None, client_id=None):

        self.dataset = dataset
        self.indices = [index for index in range(len(self.dataset))]

        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        self.id = client_id

        self.num_samples = int(len(self.dataset) / self.num_replicas)

    def __iter__(self):

        local_indices = self.indices[(self.id - 1) * self.num_samples:self.id *
                                     self.num_samples]
        assert len(local_indices) == self.num_samples
        return iter(local_indices)

    def __len__(self):
        return self.num_samples


class DictFileSampler(torch.utils.data.Sampler):
    """Partition dataset according to data_indices and client id"""
    def __init__(self, dict_file, client_id):

        data_indices = load_dict(dict_file)
        self.indices = data_indices[client_id]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
