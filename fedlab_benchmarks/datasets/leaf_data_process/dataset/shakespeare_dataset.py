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


class ShakespeareDataset(Dataset):

    def __init__(self, client_id: int, client_str: str, input: list, output: list):
        """get `Dataset` for shakespeare dataset

        Args:
            client_id (int): client id
            client_str (str): client name string
            input (list): input sentence list data
            output (list): output next-character list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.ALL_LETTERS, self.VOCAB_SIZE = self.build_vocab()
        self.data, self.targets = self.get_client_data_target(input, output)

    def build_vocab(self):
        """ according all letters to build vocab

        Vocabulary re-used from the Federated Learning for Text Generation tutorial.
        https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation

        Returns:
            all letters vocabulary list and length of vocab list
        """
        CHAR_VOCAB = list(
            'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'
        )
        ALL_LETTERS = "".join(CHAR_VOCAB)
        VOCAB_SIZE = len(ALL_LETTERS) + 4  # Vocabulary with OOV ID, zero for the padding, and BOS, EOS IDs.
        return ALL_LETTERS, VOCAB_SIZE

    def get_client_data_target(self, input: str, output: str):
        """process client data and target for input and output

        Returns: data and target for client id
        """
        data = torch.tensor(
            [self.__sentence_to_indices(sentence) for sentence in input])
        targets = torch.tensor(
            [self.__letter_to_index(letter) for letter in output])

        return data, targets

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
