.. _quickstart:

***********
Quick Start
***********

In this page, we introduce how to build a FL simulation system with FedLab in cross machine or
cross process scenario. We implement FedAvg algorithm with CNN and partitioned MNIST dataset across
clients.

Source code of this page can be seen in `fedlab-benchmarks/fedavg/cross_machine <https://github.com/SMILELab-FL/FedLab-benchmarks>`_.


Download dataset
================

FedLab provides scripts for common dataset download and partition process. Besides, FL dataset baseline
LEAF :cite:p:`caldas2018leaf` is also implemented and compatible with PyTorch interfaces.

Codes related to dataset download process are available at ``fedlab_benchamrks/datasets/data/{dataset name}``.

1. Download MNIST/CIFAR10

.. code-block:: shell-session

    $ cd fedlab_benchamrks/datasets/{mnist or cifar10}/
    $ python download_{dataset}.py

2. Partition

Run follow python file to generate partition file.

.. code-block:: shell-session

    $ python {dataset}_partition.py

Source codes of partition scripts:

.. code-block:: python

    import torchvision
    from fedlab.utils.functional import save_dict
    from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    # trainset = torchvision.datasets.MNIST(root=root, train=True, download=True)

    data_indices = noniid_slicing(trainset, num_clients=100, num_shards=200)
    save_dict(data_indices, "cifar10_noniid.pkl")

    data_indices = random_slicing(trainset, num_clients=100)
    save_dict(data_indices, "cifar10_iid.pkl")

``data_indices`` is a ``dict`` mapping from client id to data indices(list) of raw dataset.
**FedLab** provides random partition and non-I.I.D. partition methods, in which the noniid partition method is totally re-implementation in paper FedAvg.

3. LEAF dataset process

Please follow the `FedLab benchmark <https://github.com/SMILELab-FL/FedLab/tree/v1.0/fedlab_benchmarks>`_ to learn how to
generate LEAF related dataset partition.


Run FedLab demos
^^^^^^^^^^^^^^^^

**FedLab** provides both asynchronous and synchronous standard implementation demos for uses to learn. We only introduce the usage of synchronous FL system simulation demo(FedAvg) with different scenario in this page. (Code structures are similar.)

**We are very confident in the readability of FedLab code, so we recommend that users read the source code according to the following demos for better understanding.**

1. Standalone
-------------

Main process is under
`fedlab_benchamrks/algorithm/fedavg/standalone <https://github.com/SMILELab-FL/FedLab/tree/v1.0/fedlab_benchmarks/algorithm/fedavg/standalone>`_.
This is a standard usage of :class:`SerialTrainer` which allows users to simulate a group of
clients with a single process.

.. code-block:: shell-session

    $ python standalone.py --total_client 100 --com_round 10 --sample_ratio 0.1 --batch_size 10 --epochs 5 --lr 0.02 --partition iid

Run command above to start a single process simulating FedAvg algorithm with 100 clients with 10 communication round in total, with 10 clients joining each round randomly.



2. Cross-Machine
-----------------

Start a FL simulation with 1 server and 2 clients.

.. code-block:: shell-session

    $ cd fedlab_benchamrks/algorithm/fedavg/cross_machine
    $ bash quick_start.sh

The content of ``quick_start.sh`` is:

.. code-block:: shell-session

    python server.py --ip 127.0.0.1 --port 3002 --world_size 3 --dataset mnist --round 3 &
    python client.py --ip 127.0.0.1 --port 3002 --world_size 3 --rank 1 --dataset mnist &
    python client.py --ip 127.0.0.1 --port 3002 --world_size 3 --rank 2 --dataset mnist &

Cross Machine scenario allows users deploy their FL system in computer cluster. In this case, we
set the address of server as localhost. Then three process will communicate with each other
following standard FL procedure.

.. note::

    Due to the rank of torch.distributed is unique for every process. Therefore, we use rank represent client id for this scenario.


3. Scale
----------

:class:`SerialTrainer` uses less computer resources (single process) to simulate multiple clients. Cross Machine is suit for computer cluster deployment, simulating data-center FL system. In our experiment, the world size of ``torch.distributed`` can't more than 50 (Denpends on clusters), otherwise, the socket will crash, which limited the client number of FL simulation.

To improve scalability, FedLab provides scale standard implementation to combine
:class:`SerialTrainer` and :class:`ClientManager`, which allows a single process simulate multiple clients.

Source codes are available in fedlab_benchamrks/algorithm/fedavg/scale/{experiment setting name}.

Here, we take mnist-cnn as example to introduce this demo. In this demo, we set world_size=11 (1 ServerManager, 10 ClientManagers), and each ClientManager represents 10 local client dataset partition. Our data partition strategy follows the experimental setting of fedavg as well. In this way, **we only use 11 processes to simulate a FL system with 100 clients.**

To start this system, you need to open at least 2 terminal (we still use localhost as demo. Use multiple machines is OK as long as with right network configuration):

1. server (terminal 1)

.. code-block:: shell-session

    $ python server.py --ip 127.0.0.1 --port 3002 --world_size 11

2. clients (terminal 2)

.. code-block:: shell-session

    $ bash start_clt.sh 11 1 10 # launch clients from rank 1 to rank 10 with world_size 11

The content of ``start_clt.sh``:

.. code-block:: shell-session

    for ((i=$2; i<=$3; i++))
    do
    {
        echo "client ${i} started"
        python client.py --world_size $1 --rank ${i} &
        sleep 2s # wait for gpu resources allocation
    }
    done
    wait
