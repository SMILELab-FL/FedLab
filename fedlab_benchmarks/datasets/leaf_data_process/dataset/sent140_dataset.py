"""
    This is modified by [RSE-Adversarial-Defense-Github]
    https://github.com/Raibows/RSE-Adversarial-Defense/tree/de7bb5afc94d3d262cf0b08f55952800161865ce
"""

import os
import torch
from torch.utils.data import Dataset
from ..data_read_util import read_dir
from ..nlp_utils.tokenizer import Tokenizer


class Sent140Dataset(Dataset):

    def __init__(self, client_id: int, data_root: str, is_train=True, is_to_tokens=True, tokenizer=None):
        """get `Dataset` for shakespeare dataset

        Args:
            client_id (int): client id
            data_root (str): path contains train data and test data
            is_train (bool, optional): if get train data, `is_train` set True, else set False. Defaults to True
        """
        self.data_path = os.path.join(data_root, 'train') if is_train else os.path.join(data_root, 'test')
        self.client_id = client_id
        self.data, self.targets = self.get_client_data_target()
        self.data_token = []
        self.data_seq = []
        self.targets_tensor = []
        self.vocab = None
        self.tokenizer = tokenizer if tokenizer else Tokenizer('normal')
        self.maxlen = None

        if is_to_tokens:
            self.data2token()

    def get_client_data_target(self):
        """get client data for param `client_id` from `data_path`

        Returns: data and target for client id
        """
        client_id_name_dict, client_groups, client_name_data_dict = read_dir(data_dir=self.data_path)
        client_name = client_id_name_dict[self.client_id]
        data = [e[4] for e in client_name_data_dict[client_name]['x']]
        targets = client_name_data_dict[client_name]['y']

        return data, targets

    def data2token(self):
        assert self.data is not None
        for sen in self.data:
            self.data_token.append(self.tokenizer(sen))

    def token2seq(self, vocab: 'Vocab', maxlen: int):
        """transform token data to indices sequence by `vocab`

        Args:
            vocab (fedlabbenchmark.datasets.leaf_data_process_nlp_utils.vocab): vocab for data_token
            maxlen (int): max length of sentence

        Returns:
            list of integer list for data_token, and a list of tensor target
        """
        if len(self.data_seq) > 0:
            self.data_seq.clear()
            self.targets_tensor.clear()
        self.vocab = vocab
        self.maxlen = maxlen
        assert self.data_token is not None
        for tokens in self.data_token:
            self.data_seq.append(self.__encode_tokens(tokens))
        for target in self.targets:
            self.targets_tensor.append(torch.tensor(target))

    def __encode_tokens(self, tokens) -> torch.Tensor:
        """encode `maxlen` length for token_data to get indices list in `self.vocab`
        if one sentence length is shorter than maxlen, it will use pad word for padding to maxlen
        if one sentence length is longer than maxlen, it will cut the first max_words words

        Args:
            tokens (list[str]): data after tokenizer

        Returns:
            integer list of indices with `maxlen` length for tokens input
        """
        pad_word = 0
        x = [pad_word for _ in range(self.maxlen)]
        temp = tokens[:self.maxlen]
        for idx, word in enumerate(temp):
            x[idx] = self.vocab.get_index(word)
        return torch.tensor(x)

    def __len__(self):
        return len(self.targets_tensor)

    def __getitem__(self, item):
        return self.data_seq[item], self.targets_tensor[item]
