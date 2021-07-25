"""
    process femnist data in leaf json
"""

import torch
import numpy as np


def process_x(raw_x):
    """for all data in raw_x, process each 1D image to 3D image, shape is changed from (786) to (1, 28, 28)

    Args:
        raw_x (list[string]): contains a list of 1D-image to process

    Returns:
        x (list): a list of processed 3D-image data.
    """
    x = torch.from_numpy(np.asarray(raw_x))
    x = torch.reshape(x, (-1, 1, 28, 28))
    x = x.to(torch.float32)
    return x


def process_y(raw_y):
    """for all labels in raw_y, process them into torch.long

    Args:
        raw_y (list[int]): contains a list of label to process

    Returns:
        y (list[]): the processed torch.long data

    """
    y = torch.from_numpy(np.asarray(raw_y))
    y = y.to(torch.long)
    return y
