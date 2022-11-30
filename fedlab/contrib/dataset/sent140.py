import sys
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from torch.utils.data import Dataset
from .utils import Tokenizer, Vocab


class Sent140Dataset(Dataset):
    def __init__(self,
                 client_id: int,
                 client_str: str,
                 data: list,
                 targets: list,
                 is_to_tokens: bool = True,
                 tokenizer: Tokenizer = None):
        """get `Dataset` for sent140 dataset
        Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): sentence list data
            targets (list): next-character target list
            is_to_tokens (bool, optional), if tokenize data by using tokenizer
            tokenizer (Tokenizer, optional), tokenizer
        """
        self.client_id = client_id
        self.client_str = client_str
        self.data = data
        self.targets = targets
        self.data_token = []
        self.data_tokens_tensor = []
        self.targets_tensor = []
        self.vocab = None
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.fix_len = None

        self._process_data_target()
        if is_to_tokens:
            self._data2token()

    def _process_data_target(self):
        """process client's data and target
        """
        self.data = [e[4] for e in self.data]
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def _data2token(self):
        assert self.data is not None
        for sen in self.data:
            self.data_token.append(self.tokenizer(sen))

    def encode(self, vocab: 'Vocab', fix_len: int):
        """transform token data to indices sequence by `Vocab`
        Args:
            vocab (fedlab_benchmark.leaf.nlp_utils.util.vocab): vocab for data_token
            fix_len (int): max length of sentence
        Returns:
            list of integer list for data_token, and a list of tensor target
        """
        if len(self.data_tokens_tensor) > 0:
            self.data_tokens_tensor.clear()
            self.targets_tensor.clear()
        self.vocab = vocab
        self.fix_len = fix_len
        pad_idx = self.vocab.get_index('<pad>')
        assert self.data_token is not None
        for tokens in self.data_token:
            self.data_tokens_tensor.append(
                self.__encode_tokens(tokens, pad_idx))
        for target in self.targets:
            self.targets_tensor.append(torch.tensor(target))

    def __encode_tokens(self, tokens, pad_idx) -> torch.Tensor:
        """encode `fix_len` length for token_data to get indices list in `self.vocab`
        if one sentence length is shorter than fix_len, it will use pad word for padding to fix_len
        if one sentence length is longer than fix_len, it will cut the first max_words words
        Args:
            tokens (list[str]): data after tokenizer
        Returns:
            integer list of indices with `fix_len` length for tokens input
        """
        x = [pad_idx for _ in range(self.fix_len)]
        for idx, word in enumerate(tokens[:self.fix_len]):
            x[idx] = self.vocab.get_index(word)
        return torch.tensor(x)

    def __len__(self):
        return len(self.targets_tensor)

    def __getitem__(self, item):
        return self.data_tokens_tensor[item], self.targets_tensor[item]