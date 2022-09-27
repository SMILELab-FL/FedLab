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

from abc import ABC, abstractmethod

import numpy as np

from . import functional as F


class DataPartitioner(ABC):
    """Base class for data partition in federated learning.

    Examples of :class:`DataPartitioner`: :class:`BasicPartitioner`, :class:`CIFAR10Partitioner`.

    Details and tutorials of different data partition and datasets, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
    """
    def __init__(self):
        pass

    @abstractmethod
    def _perform_partition(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()


class CIFAR10Partitioner(DataPartitioner):
    """CIFAR10 data partitioner.

    Partition CIFAR10 given specific client number. Currently 6 supported partition schemes can be
    achieved by passing different combination of parameters in initialization:

    - ``balance=None``

      - ``partition="dirichlet"``: non-iid partition used in
        `Bayesian Nonparametric Federated Learning of Neural Networks <https://arxiv.org/abs/1905.12022>`_
        and `Federated Learning with Matched Averaging <https://arxiv.org/abs/2002.06440>`_. Refer
        to :func:`fedlab.utils.dataset.functional.hetero_dir_partition` for more information.

      - ``partition="shards"``: non-iid method used in FedAvg `paper <https://arxiv.org/abs/1602.05629>`_.
        Refer to :func:`fedlab.utils.dataset.functional.shards_partition` for more information.


    - ``balance=True``: "Balance" refers to FL scenario that sample numbers for different clients
      are the same. Refer to :func:`fedlab.utils.dataset.functional.balance_partition` for more
      information.

      - ``partition="iid"``: Random select samples from complete dataset given sample number for
        each client.

      - ``partition="dirichlet"``: Refer to :func:`fedlab.utils.dataset.functional.client_inner_dirichlet_partition`
        for more information.

    - ``balance=False``: "Unbalance" refers to FL scenario that sample numbers for different clients
      are different. For unbalance method, sample number for each client is drown from Log-Normal
      distribution with variance ``unbalanced_sgm``. When ``unbalanced_sgm=0``, partition is
      balanced. Refer to :func:`fedlab.utils.dataset.functional.lognormal_unbalance_partition`
      for more information. The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.

      - ``partition="iid"``: Random select samples from complete dataset given sample number for
        each client.

      - ``partition="dirichlet"``: Refer to :func:`fedlab.utils.dataset.functional.client_inner_dirichlet_partition`
        for more information.

    For detail usage, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    Args:
        targets (list or numpy.ndarray): Targets of dataset for partition. Each element is in range of [0, 1, ..., 9].
        num_clients (int): Number of clients for data partition.
        balance (bool, optional): Balanced partition over all clients or not. Default as ``True``.
        partition (str, optional): Partition type, only ``"iid"``, ``shards``, ``"dirichlet"`` are supported. Default as ``"iid"``.
        unbalance_sgm (float, optional): Log-normal distribution variance for unbalanced data partition over clients. Default as ``0`` for balanced partition.
        num_shards (int, optional): Number of shards in non-iid ``"shards"`` partition. Only works if ``partition="shards"``. Default as ``None``.
        dir_alpha (float, optional): Dirichlet distribution parameter for non-iid partition. Only works if ``partition="dirichlet"``. Default as ``None``.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.
        min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``. Only works if ``partition="noniid-labeldir"``.
        seed (int, optional): Random seed. Default as ``None``.
    """

    num_classes = 10

    def __init__(self, targets, num_clients,
                 balance=True, partition="iid",
                 unbalance_sgm=0,
                 num_shards=None,
                 dir_alpha=None,
                 verbose=True,
                 min_require_size=None,
                 seed=None):

        self.targets = np.array(targets)  # with shape (num_samples,)
        self.num_samples = self.targets.shape[0]
        self.num_clients = num_clients
        self.client_dict = dict()
        self.partition = partition
        self.balance = balance
        self.dir_alpha = dir_alpha
        self.num_shards = num_shards
        self.unbalance_sgm = unbalance_sgm
        self.verbose = verbose
        self.min_require_size = min_require_size
        # self.rng = np.random.default_rng(seed)  # rng currently not supports randint
        np.random.seed(seed)

        # partition scheme check
        if balance is None:
            assert partition in ["dirichlet", "shards"], f"When balance=None, 'partition' only " \
                                                         f"accepts 'dirichlet' and 'shards'."
        elif isinstance(balance, bool):
            assert partition in ["iid", "dirichlet"], f"When balance is bool, 'partition' only " \
                                                      f"accepts 'dirichlet' and 'iid'."
        else:
            raise ValueError(f"'balance' can only be NoneType or bool, not {type(balance)}.")

        # perform partition according to setting
        self.client_dict = self._perform_partition()
        # get sample number count for each client
        self.client_sample_count = F.samples_num_count(self.client_dict, self.num_clients)

    def _perform_partition(self):
        if self.balance is None:
            if self.partition == "dirichlet":
                client_dict = F.hetero_dir_partition(self.targets,
                                                     self.num_clients,
                                                     self.num_classes,
                                                     self.dir_alpha,
                                                     min_require_size=self.min_require_size)

            else:  # partition is 'shards'
                client_dict = F.shards_partition(self.targets, self.num_clients, self.num_shards)

        else:  # if balance is True or False
            # perform sample number balance/unbalance partition over all clients
            if self.balance is True:
                client_sample_nums = F.balance_split(self.num_clients, self.num_samples)
            else:
                client_sample_nums = F.lognormal_unbalance_split(self.num_clients,
                                                                 self.num_samples,
                                                                 self.unbalance_sgm)

            # perform iid/dirichlet partition for each client
            if self.partition == "iid":
                client_dict = F.homo_partition(client_sample_nums, self.num_samples)
            else:  # for dirichlet
                client_dict = F.client_inner_dirichlet_partition(self.targets, self.num_clients,
                                                                 self.num_classes, self.dir_alpha,
                                                                 client_sample_nums, self.verbose)

        return client_dict

    def __getitem__(self, index):
        """Obtain sample indices for client ``index``.

        Args:
            index (int): Client ID.

        Returns:
            list: List of sample indices for client ID ``index``.

        """
        return self.client_dict[index]

    def __len__(self):
        """Usually equals to number of clients."""
        return len(self.client_dict)


class CIFAR100Partitioner(CIFAR10Partitioner):
    """CIFAR100 data partitioner.

    This is a subclass of the :class:`CIFAR10Partitioner`. For details, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
    """
    num_classes = 100


class BasicPartitioner(DataPartitioner):
    """Basic data partitioner.

    Basic data partitioner, supported partition:

    - label-distribution-skew:quantity-based

    - label-distribution-skew:distributed-based (Dirichlet)

    - quantity-skew (Dirichlet)

    - IID

    For more details, please check `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_ and `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        partition (str): Partition name. Only supports ``"noniid-#label"``, ``"noniid-labeldir"``, ``"unbalance"`` and ``"iid"`` partition schemes.
        dir_alpha (float): Parameter alpha for Dirichlet distribution. Only works if ``partition="noniid-labeldir"``.
        major_classes_num (int): Number of major class for each clients. Only works if ``partition="noniid-#label"``.
        verbose (bool): Whether output intermediate information. Default as ``True``.
        min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``. Only works if ``partition="noniid-labeldir"``.
        seed (int): Random seed. Default as ``None``.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    num_classes = 2

    def __init__(self, targets, num_clients,
                 partition='iid',
                 dir_alpha=None,
                 major_classes_num=1,
                 verbose=True,
                 min_require_size=None,
                 seed=None):
        self.targets = np.array(targets)  # with shape (num_samples,)
        self.num_samples = self.targets.shape[0]
        self.num_clients = num_clients
        self.client_dict = dict()
        self.partition = partition
        self.dir_alpha = dir_alpha
        self.verbose = verbose
        self.min_require_size = min_require_size
            
        # self.rng = np.random.default_rng(seed)  # rng currently not supports randint
        np.random.seed(seed)

        if partition == "noniid-#label":
            # label-distribution-skew:quantity-based
            assert isinstance(major_classes_num, int), f"'major_classes_num' should be integer, " \
                                                       f"not {type(major_classes_num)}."
            assert major_classes_num > 0, f"'major_classes_num' should be positive."
            assert major_classes_num < self.num_classes, f"'major_classes_num' for each client " \
                                                         f"should be less than number of total " \
                                                         f"classes {self.num_classes}."
            self.major_classes_num = major_classes_num
        elif partition in ["noniid-labeldir", "unbalance"]:
            # label-distribution-skew:distributed-based (Dirichlet) and quantity-skew (Dirichlet)
            assert dir_alpha > 0, f"Parameter 'dir_alpha' for Dirichlet distribution should be " \
                                  f"positive."
        elif partition == "iid":
            # IID
            pass
        else:
            raise ValueError(
                f"tabular data partition only supports 'noniid-#label', 'noniid-labeldir', "
                f"'unbalance', 'iid'. {partition} is not supported.")

        self.client_dict = self._perform_partition()
        # get sample number count for each client
        self.client_sample_count = F.samples_num_count(self.client_dict, self.num_clients)

    def _perform_partition(self):
        if self.partition == "noniid-#label":
            # label-distribution-skew:quantity-based
            client_dict = F.label_skew_quantity_based_partition(self.targets, self.num_clients,
                                                                self.num_classes,
                                                                self.major_classes_num)

        elif self.partition == "noniid-labeldir":
            # label-distribution-skew:distributed-based (Dirichlet)
            client_dict = F.hetero_dir_partition(self.targets, self.num_clients, self.num_classes,
                                                 self.dir_alpha,
                                                 min_require_size=self.min_require_size)

        elif self.partition == "unbalance":
            # quantity-skew (Dirichlet)
            client_sample_nums = F.dirichlet_unbalance_split(self.num_clients, self.num_samples,
                                                             self.dir_alpha)
            client_dict = F.homo_partition(client_sample_nums, self.num_samples)

        else:
            # IID
            client_sample_nums = F.balance_split(self.num_clients, self.num_samples)
            client_dict = F.homo_partition(client_sample_nums, self.num_samples)

        return client_dict

    def __getitem__(self, index):
        return self.client_dict[index]

    def __len__(self):
        return len(self.client_dict)


class VisionPartitioner(BasicPartitioner):
    """Data partitioner for vision data.

    Supported partition for vision data:

    - label-distribution-skew:quantity-based

    - label-distribution-skew:distributed-based (Dirichlet)

    - quantity-skew (Dirichlet)

    - IID

    For more details, please check `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        partition (str): Partition name. Only supports ``"noniid-#label"``, ``"noniid-labeldir"``, ``"unbalance"`` and ``"iid"`` partition schemes.
        dir_alpha (float): Parameter alpha for Dirichlet distribution. Only works if ``partition="noniid-labeldir"``.
        major_classes_num (int): Number of major class for each clients. Only works if ``partition="noniid-#label"``.
        verbose (bool): Whether output intermediate information. Default as ``True``.
        seed (int): Random seed. Default as ``None``.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    num_classes = 10

    def __init__(self, targets, num_clients,
                 partition='iid',
                 dir_alpha=None,
                 major_classes_num=None,
                 verbose=True,
                 seed=None):
        super(VisionPartitioner, self).__init__(targets=targets, num_clients=num_clients,
                                                partition=partition,
                                                dir_alpha=dir_alpha,
                                                major_classes_num=major_classes_num,
                                                verbose=verbose,
                                                seed=seed)


class MNISTPartitioner(VisionPartitioner):
    """Data partitioner for MNIST.

    For details, please check :class:`VisionPartitioner`  and `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
    """
    num_features = 784


class FMNISTPartitioner(VisionPartitioner):
    """Data partitioner for FashionMNIST.

    For details, please check :class:`VisionPartitioner`  and `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_
    """
    num_features = 784


class SVHNPartitioner(VisionPartitioner):
    """Data partitioner for SVHN.

    For details, please check :class:`VisionPartitioner`  and `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_
    """
    num_features = 1024


class FCUBEPartitioner(DataPartitioner):
    """FCUBE data partitioner.

    FCUBE is a synthetic dataset for research in non-IID scenario with feature imbalance. This
    dataset and its partition methods are proposed in `Federated Learning on Non-IID Data Silos: An
    Experimental Study <https://arxiv.org/abs/2102.02079>`_.

    Supported partition methods for FCUBE:

    - feature-distribution-skew:synthetic

    - IID

    For more details, please refer to Section (IV-B-b) of original paper. For detailed usage, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    Args:
        data (numpy.ndarray): Data of dataset :class:`FCUBE`.
        partition (str): Partition type. Only supports `'synthetic'` and `'iid'`.
    """
    num_classes = 2
    num_clients = 4  # only accept partition for 4 clients

    def __init__(self, data, partition):
        if partition not in ['synthetic', 'iid']:
            raise ValueError(
                f"FCUBE only supports 'synthetic' and 'iid' partition, not {partition}.")
        self.partition = partition
        self.data = data
        if isinstance(data, np.ndarray):
            self.num_samples = data.shape[0]
        else:
            self.num_samples = len(data)

        self.client_dict = self._perform_partition()

    def _perform_partition(self):
        if self.partition == 'synthetic':
            # feature-distribution-skew:synthetic
            client_dict = F.fcube_synthetic_partition(self.data)
        else:
            # IID partition
            client_sample_nums = F.balance_split(self.num_clients, self.num_samples)
            client_dict = F.homo_partition(client_sample_nums, self.num_samples)

        return client_dict

    def __getitem__(self, index):
        return self.client_dict[index]

    def __len__(self):
        return self.num_clients


class AdultPartitioner(BasicPartitioner):
    """Data partitioner for Adult.

    For details, please check :class:`BasicPartitioner`  and `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_
    """
    num_features = 123
    num_classes = 2


class RCV1Partitioner(BasicPartitioner):
    """Data partitioner for RCV1.

    For details, please check :class:`BasicPartitioner`  and `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_
    """
    num_features = 47236
    num_classes = 2


class CovtypePartitioner(BasicPartitioner):
    """Data partitioner for Covtype.

    For details, please check :class:`BasicPartitioner`  and `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_
    """
    num_features = 54
    num_classes = 2
