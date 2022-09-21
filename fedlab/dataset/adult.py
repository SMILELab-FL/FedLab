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

import numpy as np
from sklearn.datasets import load_svmlight_file
import random
import os
from urllib.request import urlretrieve

from torch.utils.data import Dataset


class Adult(Dataset):
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
    train_file_name = "a9a"
    test_file_name = "a9a.t"
    num_classes = 2
    num_features = 123

    def __init__(self, root, train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(root):
            os.mkdir(root)

        if self.train:
            self.full_file_path = os.path.join(self.root, self.train_file_name)
        else:
            self.full_file_path = os.path.join(self.root, self.test_file_name)

        if download:
            self.download()

        if not self._local_file_existence():
            raise RuntimeError(
                f"Adult-a9a source data file not found. You can use download=True to "
                f"download it.")

        # now load from source file
        X, y = load_svmlight_file(self.full_file_path)
        X = X.todense()  # transform

        if not self.train:
            X = np.c_[X, np.zeros((len(y), 1))]  # append a zero vector at the end of X_test

        X = np.array(X, dtype=np.float32)
        y = np.array((y + 1) / 2, dtype=np.int32)  # map elements of y from {-1, 1} to {0, 1}
        print(f"Local file {self.full_file_path} loaded.")
        self.data, self.targets = X, y

    def download(self):
        if self._local_file_existence():
            print(f"Source file already downloaded.")
            return

        if self.train:
            download_url = self.url + self.train_file_name
        else:
            download_url = self.url + self.test_file_name

        urlretrieve(download_url, self.full_file_path)

    def _local_file_existence(self):
        return os.path.exists(self.full_file_path)

    def __getitem__(self, index):
        data, label = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return len(self.targets)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")
