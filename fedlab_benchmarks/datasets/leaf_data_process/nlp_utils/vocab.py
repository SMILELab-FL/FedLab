"""
    This is modified by [RSE-Adversarial-Defense-Github]
    https://github.com/Raibows/RSE-Adversarial-Defense/tree/de7bb5afc94d3d262cf0b08f55952800161865ce
"""

import os
import numpy as np


class Vocab:

    def __init__(self, origin_data_tokens, word_dim: int = 300, vocab_limit_size=80000,
                 is_using_pretrained=True, word_vec_file_path='./glove.6B.300d.txt'):
        self.file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), word_vec_file_path)
        self.word_dim = word_dim
        self.word2idx = {}
        self.word2count = {}
        self.vectors = None
        self.num = 0
        self.data_tokens = []
        self.vocab_words = []
        assert len(origin_data_tokens) > 0
        self.data_tokens = origin_data_tokens
        self.__build_words_index()
        self.__limit_dict_size(vocab_limit_size)
        if is_using_pretrained:
            print('building word vectors from {}'.format(self.file_path))
            self.__read_pretrained_word_vecs()
        print('word vectors has been built! dict size is {}'.format(self.num))

    def __build_words_index(self):
        """build word2idx and word2count for iterating sentences with a word list in data_tokens

        Returns:
            word2idx mapping word and index in local vocab, word2count mapping word and count in data_tokens list
        """
        for sen in self.data_tokens:
            for word in sen:
                if word not in self.word2idx:
                    self.word2idx[word] = self.num
                    self.word2count[word] = 1
                    self.num += 1
                else:
                    self.word2count[word] += 1

    def __limit_dict_size(self, vocab_limit_size):
        """limit word dict number by top vocab_limit_size in sorted word_count, and re-build words_vocab

        Args:
            vocab_limit_size: vocab occurrence top number

        Returns:
            get limited words_vocab list and word_dict mapping index
        """
        limit = vocab_limit_size
        self.word2count = sorted(self.word2count.items(), key=lambda x: x[1], reverse=True)
        count = 1
        self.vocab_words.append('<unk>')
        temp = {}
        for x, y in self.word2count:
            if count > limit:
                break
            temp[x] = count
            self.vocab_words.append(x)
            count += 1
        self.word2idx = temp
        self.word2idx['<unk>'] = 0
        self.num = count
        assert self.num == len(self.word2idx) == len(self.vocab_words)
        self.vectors = np.ndarray([self.num, self.word_dim], dtype='float32')

    def __read_pretrained_word_vecs(self):
        """read pretrained word vectors and get intersection with local word vector

        Returns:
            `vectors` as intersection between pretrained word2vector and local word2vector
        """
        num = 0
        word2idx_glove = {'<unk>': num}  # initial unk
        with open(self.file_path, 'r', encoding='utf-8') as file:
            file = file.readlines()
            vectors_glove = np.ndarray([len(file) + 1, self.word_dim], dtype='float32')
            vectors_glove[0] = np.random.normal(0.0, 0.3, [self.word_dim])  # unk
            for line in file:
                num += 1
                line = line.split()
                word2idx_glove[line[0]] = num
                vectors_glove[num] = np.asarray(line[-self.word_dim:], dtype='float32')

        for word, idx in self.word2idx.items():
            if idx == 0:  # unk
                continue
            if word in word2idx_glove:
                key = word2idx_glove[word]
                self.vectors[idx] = vectors_glove[key]
            else:
                self.vectors[idx] = vectors_glove[0]

    def __len__(self):
        return self.num

    def get_index(self, word: str):
        if self.word2idx.get(word) is None:
            return 0  # unknown word
        return self.word2idx[word]

    def get_word(self, index: int):
        return self.vocab_words[index]

    def get_vec(self, index: int):
        assert self.vectors is not None
        return self.vectors[index]
