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

import sys
import spacy
import re
import numpy as np
from pathlib import Path
from collections import Counter

import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))


class Tokenizer:
    """ Provide tokenizer function for a string sentence."""
    def __init__(self, tokenizer_type: str = None, is_word_level: bool = True):
        if tokenizer_type is None:
            self.tokenizer = self._split_tokenizer
        elif tokenizer_type == 'spacy':
            self.spacy = spacy.load('en_core_web_sm')
            self.tokenizer = self._spacy_tokenizer
        else:
            raise ValueError(f'Tokenizer type is error, do not have type {tokenizer_type}')
        self.token_type = tokenizer_type
        self.is_word_level = is_word_level

    def preprocess(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"<br />", "", text)
        text = re.sub(r'(\W)(?=\1)', '', text)
        text = re.sub(r"([.!?,])", r" \1", text)
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
        return text.strip()

    def _split_tokenizer(self, text: str) -> [str]:
        if self.is_word_level:
            return [tok for tok in text.split() if not tok.isspace()]
        else:
            return [tok for tok in text]

    def _spacy_tokenizer(self, text: str) -> [str]:
        if self.is_word_level:
            text = self.spacy(text)
            return [token.text for token in text if not token.text.isspace()]
        else:
            return [tok for tok in text]

    def __call__(self, text: str) -> [str]:
        text = self.preprocess(text)
        tokens = self.tokenizer(text)
        return tokens


class Vocab:
    """ Provide Vocab object for corpus from one or more datasets."""
    def __init__(self, data_tokens: list, word_dim: int=300, vocab_limit_size: int=50000, min_freq: int = 1,
                 is_using_pretrained: bool=True, vectors_path: str=None, vector_name: str='glove.6B.300d.txt',
                 specials: tuple=('<unk>', '<pad>')):
        self.data_tokens = data_tokens
        self.word2idx = {}
        self.word2count = Counter()
        self.idx2word = []  # vocab words
        self.num = 0
        self.word_dim = word_dim
        self.vectors = None
        self.specials = specials
        self.vocab_limit_size = vocab_limit_size + len(specials) if self.specials else vocab_limit_size
        self.min_freq = min_freq
        self.is_using_pretrained = is_using_pretrained
        self.vectors_path = Path(vectors_path) if vectors_path else Path(__file__).parent / "glove"
        self.vector_name = vector_name


        self._build_words_index()
        if is_using_pretrained:
            print('building word vectors from {}'.format((self.vectors_path/self.vector_name).resolve()))
            self._read_pretrained_word_vecs()
        print('word vectors has been built! dict size is {}'.format(self.num))

    def _build_words_index(self):
        """build word2idx and word2count for iterating sentences with a word list in data_tokens
        Returns:
            word2idx mapping word and index in local vocab, word2count mapping word and count in data_tokens list
        """

        for data in self.data_tokens:
            for token in data:
                self.word2count.update([token])

        # sort by frequency and limit the vocab size to generate idx2word and word2idx
        self.word2count = sorted(self.word2count.items(), key=lambda x: x[1], reverse=True)
        for idx, special in enumerate(self.specials):
            self.idx2word.append(special)  # unk-0
            self.word2idx[special] = idx
        idx = 2
        for word, count in self.word2count:
            if idx >= self.vocab_limit_size or count < self.min_freq:
                break
            self.idx2word.append(word)
            self.word2idx[word] = idx
            idx += 1
        self.num = idx

        assert self.num == len(self.word2idx) == len(self.idx2word)
        self.vectors = np.ndarray([self.num, self.word_dim], dtype='float32')

    def _read_pretrained_word_vecs(self):
        """read pretrained word vectors and get intersection with local word vector
    
        Returns:
            `vectors` as intersection between pretrained word2vector and local word2vector
        """
        vector_file_path = self.vectors_path / self.vector_name
        word2idx_glove = {'<unk>': 0, '<pad>': 1}  # initial unk, pad
        idx = 2
        with open(vector_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            vectors_glove = np.ndarray([len(lines) + 2, self.word_dim], dtype='float32')
            vectors_glove[0] = np.random.normal(0.0, 0.3, [self.word_dim])  # unk
            vectors_glove[1] = torch.zeros(self.word_dim)  # pad

            for line in lines:
                line = line.split()
                word2idx_glove[line[0]] = idx
                vectors_glove[idx] = np.asarray(line[-self.word_dim:], dtype='float32')
                idx += 1

        # load vector for local word vocab
        for word, idx in self.word2idx.items():
            if idx == 0:  # unk
                continue
            if word in word2idx_glove:
                key = word2idx_glove[word]
                self.vectors[idx] = vectors_glove[key]
            else:
                self.vectors[idx] = vectors_glove[0]

        self.vectors = torch.tensor(self.vectors)
    def __len__(self):
        return self.num

    def get_index(self, word: str):
        if self.word2idx.get(word) is None:
            return 0  # unknown word
        return self.word2idx[word]

    def get_word(self, index: int):
        return self.idx2word[index]

    def get_vec(self, index: int):
        assert self.vectors is not None
        return self.vectors[index]
