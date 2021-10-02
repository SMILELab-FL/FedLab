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
import random
import os
import pickle

from torch.utils.data import Dataset

from ..functional import save_dict, load_dict


class FCUBE(Dataset):
    """FCUBE data set.

    From paper `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_.

    Args:
        root (str): Root for data file.
        train (bool, optional): Training set or test set. Default as ``True``.
        generate (bool, optional): Whether to generate synthetic dataset. If ``True``, then generate new synthetic FCUBE data even existed. Default as ``True``.
        num_samples (int, optional): Total number of samples to generate. We suggest to use 4000 for training set, and 1000 for test set. Default is ``4000`` for trainset.
    """
    train_files = {'data': "fcube_train_X", 'targets': "fcube_train_y"}
    test_files = {'data': "fcube_test_X", 'targets': "fcube_test_y"}
    num_clients = 4  # only for 4 clients

    def __init__(self, root, train=True, generate=True, num_samples=4000):
        self.data = None
        self.targets = None
        self.train = train
        self.generate = generate
        self.num_samples = num_samples

        if not os.path.exists(root):
            os.makedirs(root)

        files = self.train_files if train else self.test_files
        files = {key: files[key] + f"_{num_samples}.npy" for key in files}

        full_file_paths = {key: os.path.join(root, files[key]) for key in files}
        self.full_file_paths = full_file_paths

        if generate is False:
            if os.path.exists(full_file_paths['data']) and os.path.exists(
                    full_file_paths['targets']):
                print(
                    f"FCUBE data already generated. Load from file {full_file_paths['data']} "
                    f"and {full_file_paths['targets']}...")
                self.data = np.load(full_file_paths['data'])
                self.targets = np.load(full_file_paths['targets'])
                print(f"FCUBE data loaded from local file.")
            else:
                raise RuntimeError(
                    f"FCUBE data file not found. You can use generate=True to generate it.")
        else:
            # Generate file by force
            print("Generate FCUBE data now...")
            if train:
                self._generate_train()
            else:
                self._generate_test()

            self._save_data()  # save to local npy files

    def _generate_train(self):
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(int(self.num_samples / 4)):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 == 1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)

        self.data = np.array(X_train, dtype=np.float32)
        self.targets = np.array(y_train, dtype=np.int32)

    def _generate_test(self):
        X_test, y_test = [], []
        for i in range(self.num_samples):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1 > 0:
                y_test.append(0)
            else:
                y_test.append(1)

        self.data = np.array(X_test, dtype=np.float32)
        self.targets = np.array(y_test, dtype=np.int64)

    def _save_data(self):
        np.save(self.full_file_paths['data'], self.data)
        print(f"{self.full_file_paths['data']} generated.")
        np.save(self.full_file_paths['targets'], self.targets)
        print(f"{self.full_file_paths['targets']} generated.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (features, target) where target is index of the target class.
        """
        return self.data[index], self.targets[index]
