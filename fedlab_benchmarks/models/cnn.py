"""CNN model in pytorch
References:
    [1] Reddi S, Charles Z, Zaheer M, et al.
    Adaptive Federated Optimization. ICML 2020.
    https://arxiv.org/pdf/2003.00295.pdf
"""

import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=28):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz // 4
        self.n_feat = 50 * feat_map_sz * feat_map_sz

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 20, 5, padding=2),
            # nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            # nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.linear = nn.Sequential(
            nn.Linear(self.n_feat, 500),
            # nn.BatchNorm1d(500),
            #            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(
            500, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class CNN_Femnist(nn.Module):
    """Used for EMNIST experiments in references[1]
    Args:
        only_digits (bool, optional): If True, uses a final layer with 10 outputs, for use with the
            digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
            If selfalse, uses 62 outputs for selfederated Extended MNIST (selfEMNIST)
            EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373
            Defaluts to `True`
    Returns:
        A `torch.nn.Module`.
    """
    def __init__(self, only_digits=True):
        super(CNN_Femnist, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        # x = self.softmax(x)
        return x


class CNN_Mnist(nn.Module):
    def __init__(self):
        super(CNN_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x