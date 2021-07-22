import torch
import numpy as np


def process_x(raw_x):
    """for all word strings in raw_x, process each char to index in ALL_LETTERS

    Args:
        raw_x (list[string]): contains a list of word strings to process

    Returns:
        x (list[list]): int indices list for words in raw_x in ALL_LETTERS
    Return:
        len(vocab) by len(raw_x_batch) np array
    """
    x = torch.from_numpy(np.asarray(raw_x))
    x = torch.reshape(x, (-1, 1, 28, 28))
    x = x.to(torch.float32)
    return x


def process_y(raw_y):
    """for all labels(next char) in raw_y, process them to index in ALL_LETTERS

    Args:
        raw_y (list[char]): contains a list of label to process

    Returns:
        y (list[int]): the index for all chars in raw_y in ALL_LETTERS list

    """
    y = torch.from_numpy(np.asarray(raw_y))
    y = y.to(torch.long)
    return y
