"""
    process shakespeare data in leaf json
"""

import torch
import numpy as np
from fedlab_benchmarks.datasets.leaf_data_process.shakespeare.language_utils import word_to_indices, letter_to_index


def process_x(raw_x):
    """for all word strings in raw_x, process each char to index in ALL_LETTERS

    Args:
        raw_x (list[string]): contains a list of word strings to process

    Returns:
        x (list[list]): int indices list for words in raw_x in ALL_LETTERS
    """
    x = [word_to_indices(word) for word in raw_x]
    x = torch.from_numpy(np.asarray(x))
    return x


def process_y(raw_y):
    """for all labels(next char) in raw_y, process them to index in ALL_LETTERS

    Args:
        raw_y (list[char]): contains a list of label to process

    Returns:
        y (list[int]): the index for all chars in raw_y in ALL_LETTERS list
    """
    y = [letter_to_index(c) for c in raw_y]
    y = torch.from_numpy(np.asarray(y))
    return y
