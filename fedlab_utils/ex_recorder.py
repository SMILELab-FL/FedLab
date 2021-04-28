# unfinished

import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')


class ExRecorder_ClassifyTask:
    """Provide essential functions to record experiment """

    def __init__(self, filename):
        self._title = ["cross_entropy", "accuracy", "top5_accuracy"]
        self._filename = filename
        self._records = []

        self._acc_list = []
        self._CE_loss_list = []
        self._top5acc_list = []

    def add_record(self, record):
        """Add a single record

        Args:
            record (dict): {"cross_entropy": data, "accuracy": data, "top5_accuracy":data}

        Raises:

        """
        self._acc_list.append(record['accuracy'])
        self._CE_loss_list.append(record['cross_entropy'])
        self._top5acc_list.append(record['top5_accuracy'])

    def add_log_direct(self, record):
        """explaintion

        Args:

        Returns:

        Raises:

        """
        temp = []
        for key in record:
            temp.append(record[key])
        self._records.append(temp)

    def save_to_file(self):
        """explaintion

        Args:

        Returns:

        Raises:

        """
        with open(self._filename, "w") as f:
            for item in self._records:
                f.write(str(item) + "\n")

    def draw(self, path, accuracy_list, loss_list,):
        E = np.arange(1, len(accuracy_list)+1)
        plt.subplot(121)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(E, accuracy_list, label="accuracy")

        E = np.arange(1, len(loss_list)+1)
        plt.subplot(122)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(E, loss_list, label="loss")

        plt.tight_layout()
        plt.savefig(os.path.join(path, "local_fig.jpg"))
        plt.clf()


class ExRecorder(object):
    """Record information during training"""

    def __init__(self, filename):
        self._title = ["cross_entropy", "accuracy", "top5_accuracy"]
        self._filename = filename
        self._records = []

    def add_log(self, record):
        # print(record)
        temp = []
        for key in record:
            temp.append(record[key].detach().item())
        self._records.append(temp)

    def add_log_direct(self, record):
        # print(record)
        temp = []
        for key in record:
            temp.append(record[key])
        self._records.append(temp)

    def save_to_file(self):
        with open(self._filename, "w") as f:
            for item in self._records:
                f.write(str(item) + "\n")
