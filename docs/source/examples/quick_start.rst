.. _quickstart:

***********
Quick Start
***********

In this page, we introduce the provided quick start demos. And the start scripts for FL simulation system with FedLab in different scenario. We implement FedAvg algorithm with MLP network and partitioned MNIST dataset across clients.

Source code  can be seen in `fedlab/examples/ <https://github.com/SMILELab-FL/FedLab/tree/master/examples>`_.


Download dataset
================

FedLab provides scripts for common dataset download and partition process. Besides, FL dataset baseline
LEAF :cite:p:`caldas2018leaf` is also implemented and compatible with PyTorch interfaces.

Codes related to dataset download process are available at ``fedlab_benchamrks/datasets/{dataset name}``.

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

Source code is under
`fedlab/examples/standalone-mnist <https://github.com/SMILELab-FL/FedLab/tree/master/examples/standalone-mnist>`_.
This is a standard usage of :class:`SerialTrainer` which allows users to simulate a group of
clients with a single process.

.. code-block:: shell-session

    $ python standalone.py --total_client 100 --com_round 3 --sample_ratio 0.1 --batch_size 100 --epochs 5 --lr 0.02

or

.. code-block:: shell-session

    $ bash launch_eg.sh

Run command above to start a single process simulating FedAvg algorithm with 100 clients with 10 communication round in total, with 10 clients sampled randomly at each round .


2. Cross-process
-----------------

Source code is under `fedlab/examples/cross-process-mnist <https://github.com/SMILELab-FL/FedLab/tree/master/examples/cross-process-mnist>`_ 

Start a FL simulation with 1 server and 2 clients.

.. code-block:: shell-session

    $ bash launch_eg.sh

The content of ``launch_eg.sh`` is:

.. code-block:: shell-session

    python server.py --ip 127.0.0.1 --port 3001 --world_size 3 --round 3 &

    python client.py --ip 127.0.0.1 --port 3001 --world_size 3 --rank 1 &

    python client.py --ip 127.0.0.1 --port 3001 --world_size 3 --rank 2  &

    wait

Cross-process scenario allows users deploy their FL system in computer cluster. Although in this case, we
set the address of server as localhost. Then three process will communicate with each other
following standard FL procedure.

.. note::

    Due to the rank of torch.distributed is unique for every process. Therefore, we use rank represent client id in this scenario.


3. Cross-process with SerialTrainer
------------------------------------

:class:`SerialTrainer` uses less computer resources (single process) to simulate multiple clients. Cross-pross is suit for computer cluster deployment, simulating data-center FL system. In our experiment, the world size of ``torch.distributed`` can't more than 50 (Denpends on clusters), otherwise, the socket will crash, which limited the client number of FL simulation.

To improve scalability, FedLab provides scale standard implementation to combine
:class:`SerialTrainer` and :class:`ClientManager`, which allows a single process simulate multiple clients.

Source codes are available in `fedlab_benchamrks/algorithm/fedavg/scale/{experiment setting name} <https://github.com/SMILELab-FL/FedLab-benchmarks/tree/master/fedlab_benchmarks/fedavg/scale>`_.

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


4. Hierachical
------------------------------------

**Hierarchical** mode for **FedLab** is designed for situation tasks on multiple computer clusters (in different LAN) or the real-world scenes. To enable the inter-connection for different computer clusters, **FedLab** develops ``Scheduler`` as middle-server process to connect client groups. Each ``Scheduler`` manages the communication between the global server and clients in a client group. And server can communicate with clients in different LAN via corresponding ``Scheduler``. The computation mode of a client group for each scheduler can be either **standalone** or **cross-process**.

The demo of Hierachical with hybrid client (standalone and serial trainer) is given in `fedlab/examples/hierarchical-hybrid-mnist <https://github.com/SMILELab-FL/FedLab/tree/master/examples/hierarchical-hybrid-mnist>`_.

Run all scripts together:

.. code-block:: shell-session

    $ bash launch_eg.sh

Run scripts seperately:

.. code-block:: shell-session

    # Top server in terminal 1
    $ bash launch_topserver_eg.sh

    # Scheduler1 + Ordinary trainer with 1 client + Serial trainer with 10 clients in terminal 2:
    bash launch_cgroup1_eg.sh

    # Scheduler2 + Ordinary trainer with 1 client + Serial trainer with 10 clients in terminal 3:
    $ bash launch_cgroup2_eg.sh
