import os
import torch
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, client_id: int, client_str: str, data: list,
                 targets: list):
        """get `Dataset` for shakespeare dataset
        Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): sentence list data
            targets (list): next-character target list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.ALL_LETTERS, self.VOCAB_SIZE = self._build_vocab()
        self.data = data
        self.targets = targets
        self._process_data_target()

    def _build_vocab(self):
        """ according all letters to build vocab
        Vocabulary re-used from the Federated Learning for Text Generation tutorial.
        https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
        Returns:
            all letters vocabulary list and length of vocab list
        """
        ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        VOCAB_SIZE = len(ALL_LETTERS)
        return ALL_LETTERS, VOCAB_SIZE

    def _process_data_target(self):
        """process client's data and target
        """
        self.data = torch.tensor(
            [self.__sentence_to_indices(sentence) for sentence in self.data])
        self.targets = torch.tensor(
            [self.__letter_to_index(letter) for letter in self.targets])

    def __sentence_to_indices(self, sentence: str):
        """Returns list of integer for character indices in ALL_LETTERS
        Args:
            sentence (str): input sentence
        Returns: a integer list of character indices
        """
        indices = []
        for c in sentence:
            indices.append(self.ALL_LETTERS.find(c))
        return indices

    def __letter_to_index(self, letter: str):
        """Returns index in ALL_LETTERS of given letter
        Args:
            letter (char/str[0]): input letter
        Returns: int index of input letter
        """
        index = self.ALL_LETTERS.find(letter)
        return index

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
        