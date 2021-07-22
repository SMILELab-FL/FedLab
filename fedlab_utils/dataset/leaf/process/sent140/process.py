import torch
import numpy as np

from fedlab_utils.dataset.leaf.process.sent140.language_utils import bag_of_words, val_to_vec, get_word_emb_arr

import os
print(os.path.abspath(os.curdir))

# _, _, VOCAB = get_word_emb_arr("embs.json")
# vocab_size = len(VOCAB)


def process_x(raw_x):
    """for all word strings in raw_x, process each string to get bag of words representation in VOCAB

    Args:
        raw_x (list[string]): contains a list of word strings to process

    Returns:
        x (list[list]): int indices list for words in raw_x in ALL_LETTERS
    Return:
        list contains integer lists, which record the number of each word for one string
    """

    x = [e[4] for e in raw_x]  # list of lines/phrases
    x = [bag_of_words(line, VOCAB) for line in x]
    x = torch.from_numpy(np.array(x))
    return x


def process_y(raw_y):
    """for all labels(class) in raw_y, process them to one-hot vector

    Args:
        raw_y (list): contains a list of label to process

    Returns:
        y (list): list contains one-hot vectors for raw_y's labels

    """
    y = [int(e) for e in raw_y]
    y = [val_to_vec(2, e) for e in y]
    y = torch.from_numpy(np.array(y))
    return y
