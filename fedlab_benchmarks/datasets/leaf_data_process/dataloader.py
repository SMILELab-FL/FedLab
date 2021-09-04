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

"""
    get dataloader for dataset in LEAF processed
"""

import os
import torch
import pickle
from torch.utils.data import ConcatDataset
from .nlp_utils.dataset_vocab.sample_build_vocab import get_built_vocab


def get_dataset_pickle(dataset_name, client_id, dataset_type, path=None):
    """load dataset file for `dataset` based on client with client_id
        Args:
            dataset_name (str): string of dataset name to get vocab
            client_id (int): client id
            dataset_type (str): Dataset type {train, test}
            path (str, optional): one specific pickle file path, if given, use it to get dataset firstly
        Returns:
            if there is no built pickle file for `dataset`, return None, else return responding dataset
        """
    if path is not None:
        file_path = path
    else:
        file_path = "../data/{}/data/pickle_dataset/{}/{}_{}.pickle".format(dataset_name, dataset_type,
                                                                            dataset_type, client_id)
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), file_path)
    if not os.path.exists(file_path):
        print('There is no built dataset file for {}, please run `create_datasets_and_save.sh` in leaf firstly.'
              .format(dataset_name))
        return None
    dataset = pickle.load(open(file_path, 'rb'))
    return dataset


def get_dataset_pickle_by_path(file_path: str):
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), file_path)
    if not os.path.exists(file_path):
        print('There is no built dataset file for {}.'.format(file_path))
        return None
    dataset = pickle.load(open(file_path, 'rb'))
    return dataset


def get_LEAF_dataloader(dataset: str, client_id=0, batch_size=128):
    """Get dataloader with ``batch_size`` param for client with ``client_id``

    Args:
        dataset (str): dataloader for dataset
        client_id (int, optional): assigned client_id to get dataloader for this client. Defaults to 0
        batch_size (int, optional): the number of batch size for dataloader. Defaults to 128

    Returns:
        A tuple with train dataloader and test dataloader for the client with `client_id`

    Examples:
        trainloader, testloader = get_LEAF_dataloader(dataset='femnist', client_id=args.local_rank - 1)
    """
    trainset = get_dataset_pickle(dataset_name=dataset, client_id=client_id, dataset_type='train')
    testset = get_dataset_pickle(dataset_name=dataset, client_id=client_id, dataset_type='test')

    if dataset == 'sent140':
        vocab = get_built_vocab(dataset)
        trainset.token2seq(vocab, maxlen=300)
        testset.token2seq(vocab, maxlen=300)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        drop_last=False)  # avoid train dataloader size 0
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=len(testset),
        drop_last=False,
        shuffle=False)
    
    return trainloader, testloader


def get_LEAF_all_test_dataloader(dataset: str):
    """Get dataloader for all clients' test pickle file

    Args:
        dataset (str): dataset name

    Returns:
        ConcatDataset for all clients' test dataset
    """
    test_pickle_files_dir = "../data/{}/data/pickle_dataset/test".format(dataset)
    testset_list = []
    for file in os.listdir(test_pickle_files_dir):
        testset = get_dataset_pickle(dataset, client_id=None, dataset_type='test', path=file)
        testset_list.append(testset)
    all_testset = ConcatDataset(testset_list)
    return all_testset
