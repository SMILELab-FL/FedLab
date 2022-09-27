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


class Covtype(Dataset):
    num_classes = 2
    num_features = 54
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2"
    source_file_name = "covtype.libsvm.binary.bz2"

    def __init__(self, root,
                 train=True,
                 train_ratio=0.75,
                 transform=None,
                 target_transform=None,
                 download=False,
                 generate=False,
                 seed=None):
        self.root = root
        self.train = train
        self.train_ratio = train_ratio
        self.transform = transform
        self.target_transform = target_transform
        np.random.seed(seed)

        if not os.path.exists(root):
            os.mkdir(root)

        self.full_file_name = {
            'train': {'data': os.path.join(root, f"covtype_train_X_{train_ratio:.2f}.npy"),
                      'targets': os.path.join(root, f"covtype_train_y_{train_ratio:.2f}.npy")},
            'test': {
                'data': os.path.join(root, f"covtype_test_X_{1 - train_ratio:.2f}.npy"),
                'targets': os.path.join(root, f"covtype_test_y_{1 - train_ratio:.2f}.npy")}}

        self.full_source_file_name = os.path.join(root, self.source_file_name)

        if download:
            self.download()

        if generate:
            self.generate()

        if not self._local_npy_existence():
            raise RuntimeError(
                f"{self.full_source_file_name} do not exist. You can set download=True and "
                f"generate=True to generate local data split.")
        else:
            print(f"npy files already existed.")

        # load train/test split from local npy files
        if self.train:
            self.data = np.load(self.full_file_name['train']['data'])
            self.targets = np.load(self.full_file_name['train']['targets'])
        else:
            self.data = np.load(self.full_file_name['test']['data'])
            self.targets = np.load(self.full_file_name['test']['targets'])

    def download(self):
        if self._local_source_file_existence():
            print(f"Source file already existed: {self.full_source_file_name}")
            return

        print(f"Try to down load {self.full_source_file_name} from {self.url} ...")
        urlretrieve(self.url, self.full_source_file_name)
        print(f"Source file download done.")

    def generate(self):
        if self._local_npy_existence():
            print(f"npy files already existed: ")
            print(f"train: {self.full_file_name['train']}")
            print(f"test: {self.full_file_name['test']}")
            return

        if not self._local_source_file_existence():
            raise RuntimeError(
                f"{self.full_source_file_name} do not exist. You can set download=True to download "
                f"source file first.")

        print(f"Load original train set from {self.full_source_file_name}.")
        orig_X_train, orig_y_train = load_svmlight_file(self.full_source_file_name)
        orig_X_train = orig_X_train.todense()

        # split original train data into customized train/test split
        print(
            f"Split original train set into {self.train_ratio:.2f}train:{1 - self.train_ratio:.2f}test ...")
        num_total = orig_X_train.shape[0]
        num_train = int(num_total * self.train_ratio)
        orig_y_train = orig_y_train - 1
        idxs = np.random.permutation(num_total)
        X_train = np.array(orig_X_train[idxs[:num_train]], dtype=np.float32)
        y_train = np.array(orig_y_train[idxs[:num_train]], dtype=np.int32)
        X_test = np.array(orig_X_train[idxs[num_train:]], dtype=np.float32)
        y_test = np.array(orig_y_train[idxs[num_train:]], dtype=np.int32)
        print(f"Train/test split done.")

        np.save(self.full_file_name['train']['data'], X_train)
        np.save(self.full_file_name['train']['targets'], y_train)
        np.save(self.full_file_name['test']['data'], X_test)
        np.save(self.full_file_name['test']['targets'], y_test)
        print(f"Train/test save done.")

    def _local_npy_existence(self):
        npy_existence = [os.path.exists(self.full_file_name['train']['data']),
                         os.path.exists(self.full_file_name['train']['targets']),
                         os.path.exists(self.full_file_name['test']['data']),
                         os.path.exists(self.full_file_name['test']['targets'])]
        return all(npy_existence)

    def _local_source_file_existence(self):
        return os.path.exists(self.full_source_file_name)

    def __getitem__(self, index):
        data, label = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return len(self.targets)
