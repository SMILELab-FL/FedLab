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

import os
import torch
from torch.utils.data import Dataset
from ..data_read_util import read_dir


class FemnistDataset(Dataset):

    def __init__(self, client_id: int, data_root: str, is_train: bool):
        """get `Dataset` for femnist dataset

        Args:
            client_id (int): client id
            data_root (str): path contains train data and test data
            is_train (bool): if get train data, `is_train` set True, else set False
        """
        self.data_path = os.path.join(data_root, 'train') if is_train else os.path.join(data_root, 'test')
        self.client_id = client_id
        self.data, self.targets = self.get_client_data_target()

    def get_client_data_target(self):
        """get client data for param `client_id` from `data_path`

        Returns: data and target for client id
        """
        client_id_name_dict, client_groups, client_name_data_dict = read_dir(data_dir=self.data_path)
        client_name = client_id_name_dict[self.client_id]
        data = torch.tensor(client_name_data_dict[client_name]['x'], dtype=torch.float32)
        data = torch.reshape(data, (-1, 1, 28, 28))
        targets = torch.tensor(client_name_data_dict[client_name]['y'], dtype=torch.long)
        return data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
