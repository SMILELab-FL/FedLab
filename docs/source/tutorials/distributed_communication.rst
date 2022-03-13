.. _distributed-communication:

*************************
Distributed Communication
*************************


How to initialize distributed network?
======================================

FedLab uses `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ as point-to-point communication tools. The communication backend is Gloo as default. FedLab processes send/receive data through TCP network connection. Here is the details of how to initialize the distributed network. 

You need to assign right ethernet to :class:`DistNetwork`, making sure ``torch.distributed`` network initialization works. :class:`DistNetwork` is for quickly network configuration, which you can create one as follows:

.. code-block:: python

    from fedlab.core.network import DistNetwork
    world_size = 10
    rank = 0  # 0 for server, other rank for clients
    ethernet = None
    server_ip = '127.0.0.1'
    server_port = 1234
    network = DistNetwork(address=(server_ip, server_port), world_size, rank, ethernet)

    network.init_network_connection() # call this method to start connection.
    network.close_network_connection() # call this method to shutdown connection.

- The ``(server_ip, server_port)`` is the address of server. please be aware of that the rank of server is 0 as default.
- Make sure ``world_size`` is the same across process.
- Rank should be different (from ``0`` to ``world_size-1``).
- world_size = 1 (server) + client number.
- The ethernet is None as default. torch.distributed will try finding the right ethernet automatically. 
- The ``ethernet_name`` must be checked (using ``ifconfig``). Otherwise, network initialization would fail.

If the automatically detected interface does not work, users are required to assign a right network interface for Gloo, by assigning in code or setting the environment variables ``GLOO_SOCKET_IFNAME``, for example ``export GLOO_SOCKET_IFNAME=eth0`` or ``os.environ['GLOO_SOCKET_IFNAME'] = "eth0"``.

.. note::

    Check the available ethernet:

    .. code-block:: shell-session

        $ ifconfig

How to create package?
======================

The basic communication unit in FedLab is called package. The communication module of FedLab is in fedlab/core/communicator. :class:`Package` defines the basic data structure of network package. It contains header and content. 

.. code-block:: python

    p = Package()
    p.header   # A tensor with size = (5,).
    p.content  # A tensor with size = (x,).

Currently, you can create a network package from following methods:

.. note::

    Currently, following restrictions need to be noticed：
        1. **Tensor shape:** **FedLab** only supports vectorized tensors as content, which means that tensors with different shape should be flatterned before appended into Package (call tensor.view(-1)).
        2. **Data type:** Package doesn't accept tensors of different data type. In other words, **FedLab** force all appended tensors to be the same data type as the first appended tensor. Torch data types like **[torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]** are supported.

1. initialize with tensor

.. code-block:: python

    tensor = torch.Tensor(size=(10,))
    package = Package(content=tensor)

2. initialize with tensor list

.. code-block:: python

    tensor_sizes = [10, 5, 8]
    tensor_list = [torch.rand(size) for size in tensor_sizes]
    package = Package(content=tensor_list)

3. append a tensor to exist package

.. code-block:: python

    tensor = torch.Tensor(size=(10,))
    package = Package(content=tensor)

    new_tensor = torch.Tensor(size=(8,))
    package.append_tensor(new_tensor)

4. append a tensor list to exist package

.. code-block:: python

    tensor_sizes = [10, 5, 8]
    tensor_list = [torch.rand(size) for size in tensor_sizes]

    package = Package()
    package.append_tensor_list(tensor_list)

Two static methods are provided by Package to parse header and content:

.. code-block:: python

    p = Package()
    Package.parse_header(p.header)  # necessary information to describe the package
    Package.parse_content(p.slices, p.content) # tensor list associated with the tensor sequence appended into.

How to send package?
====================================

The point-to-point communicating agreements is implemented in PackageProcessor module. PackageProcessor is a static class to manage package sending/receiving procedure. 

User can send a package to a process with rank=0 (the parameter dst must be assigned):

.. code-block:: python

    p = Package()
    PackageProcessor.send_package(package=p, dst=0)

or, receive a package from rank=0 (set the parameter src=None to receive package from any other process):

.. code-block:: python

    sender_rank, message_code, content = PackageProcessor.recv_package(src=0)


Point-to-point communication
=============================

In recent update, we hide the communication details from user and provide simple APIs. :class:`DistNetwork` now provies two basic communication APIs: :meth:`send()` and :meth:`recv()`.

.. code-block:: python

    def send(self, content=None, message_code=None, dst=0):
        """Send tensor to process rank=dst"""
        pack = Package(message_code=message_code, content=content)
        PackageProcessor.send_package(pack, dst=dst)

    def recv(self, src=None):
        """Receive tensor from process rank=src"""
        sender_rank, message_code, content = PackageProcessor.recv_package(
            src=src)
        return sender_rank, message_code, content
