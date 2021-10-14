.. _dataset-partition:

*************************************
Federated Dataset and DataPartitioner
*************************************

Sophisticated in real world, FL need to handle various kind of data distribution scenarios, including
iid and non-iid scenarios. Though there already exists some datasets and partition schemes for published data benchmark,
it still can be very messy and hard for researchers to partition datasets according to their specific
research problems, and maintain partition results during simulation. FedLab provides :class:`fedlab.utils.dataset.partition.DataPartitioner` that allows you to use pre-partitioned datasets as well as your own data. :class:`DataPartitioner` stores sample indices for each client given a data partition scheme. Also, FedLab provides some extra datasets that are used in current FL researches while not provided by official Pytorch :class:`torchvision.datasets` yet.

.. note::

    Current implementation and design of this part are based on  LEAF :cite:p:`caldas2018leaf`, :cite:t:`acar2020federated`, :cite:t:`yurochkin2019bayesian` and NIID-Bench :cite:p:`li2021federated`.

Vision Data
===========

CIFAR10
^^^^^^^

FedLab provides a number of pre-defined partition schemes for some datasets (such as CIFAR10) that subclass :class:`fedlab.utils.dataset.partition.DataPartitioner` and implement functions specific to particular partition scheme. They can be used to prototype and benchmark your FL algorithms.

Tutorial for :class:`CIFAR10Partitioner`: :ref:`CIFAR10 tutorial <data-cifar10>`.


CIFAR100
^^^^^^^^

Notebook tutorial for :class:`CIFAR100Partitioner`: `CIFAR100 tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/blob/master/fedlab_benchmarks/datasets/cifar100/data_partitioner.ipynb>`_.



FMNIST
^^^^^^

Notebook tutorial for data partition of FMNIST (FashionMNIST) : `FMNIST tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/blob/master/fedlab_benchmarks/datasets/fmnist/fmnist_tutorial.ipynb>`_.


MNIST
^^^^^

MNIST is very similar with FMNIST, please check `FMNIST tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/blob/master/fedlab_benchmarks/datasets/fmnist/fmnist_tutorial.ipynb>`_.

SVHN
^^^^

Data partition tutorial for SVHN: `SVHN tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/blob/master/fedlab_benchmarks/datasets/svhn/svhn_tutorial.ipynb>`_

CelebA
^^^^^^

Data partition for CelebA: `CelebA tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/tree/master/fedlab_benchmarks/datasets/celeba>`_.



FEMNIST
^^^^^^^

Data partition of FEMNIST: `FEMNIST tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/tree/master/fedlab_benchmarks/datasets/femnist>`_.



Text Data
=========

Shakespeare
^^^^^^^^^^^

Data partition of Shakespeare dataset: `Shakespeare tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/tree/master/fedlab_benchmarks/datasets/shakespeare>`_.


Sent140
^^^^^^^

Data partition of Sent140: `Sent140 tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/tree/master/fedlab_benchmarks/datasets/sent140>`_.

Reddit
^^^^^^
Data partition of Reddit: `Reddit tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/tree/master/fedlab_benchmarks/datasets/reddit>`_.


Tabular Data
============

Adult
^^^^^

Adult is from `LIBSVM Data <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html>`_. Its original source is from `UCI <http://archive.ics.uci.edu/ml/index.php>`_/Adult. FedLab provides both ``Dataset`` and :class:`DataPartitioner` for Adult. Notebook tutorial for Adult: `Adult tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/blob/master/fedlab_benchmarks/datasets/adult/adult_tutorial.ipynb>`_.


Covtype
^^^^^^^

Covtype is from `LIBSVM Data <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html>`_. Its original source is from `UCI <http://archive.ics.uci.edu/ml/index.php>`_/Covtype. FedLab provides both ``Dataset`` and :class:`DataPartitioner` for Covtype. Notebook tutorial for Covtype: `Covtype tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/blob/master/fedlab_benchmarks/datasets/covtype/covtype_tutorial.ipynb>`_.


RCV1
^^^^

RCV1 is from `LIBSVM Data <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html>`_. Its original source is from `UCI <http://archive.ics.uci.edu/ml/index.php>`_/RCV1. FedLab provides both ``Dataset`` and :class:`DataPartitioner` for RCV1. Notebook tutorial for RCV1: `RCV1 tutorial <https://github.com/SMILELab-FL/FedLab-benchmarks/blob/master/fedlab_benchmarks/datasets/rcv1/rcv1_tutorial.ipynb>`_.


Synthetic Data
==============

FCUBE
^^^^^

FCUBE is a synthetic dataset for federated learning. FedLab provides both ``Dataset`` and :class:`DataPartitioner` for FCUBE. Tutorial for FCUBE: :ref:`FCUBE tutorial <fcube-tutorial>`.


LEAF-Synthetic
^^^^^^^^^^^^^^

LEAF-Synthetic is a federated dataset proposed by LEAF. Client number, class number and feature dimensions can all be customized by user.

Please check `LEAF-Synthetic <https://github.com/SMILELab-FL/FedLab-benchmarks/tree/master/fedlab_benchmarks/datasets/synthetic>`_ for more details.
