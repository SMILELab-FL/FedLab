**************************
Communication Strategy
**************************
Communication strategy is implemented by ClientManager and ServerManager together.

The prototype of ``NetworkManager`` is defined in ``fedlab.core.network_manager``, which is also a subclass of ``torch.multiprocessing.process``.    

Typically, standard implementations is shown in ``fedlab.core.client.manager`` and ``fedlab.core.server.manager``.``NetworkManager`` manages network operation and control flow procedure.

Base class definition shows below:

.. code-block:: python

    class NetworkManager(Process):
        """Abstract class

        Args:
            handler (ClientTrainer or ParameterServerBackendHandler, optional): Backend computation handler for client or server.
            newtork (DistNetwork): object to manage torch.distributed network communication.
        """
        def __init__(self, network, handler=None):
            super(NetworkManager, self).__init__()

            self._handler = handler
            self._network = network

        def run(self):
            raise NotImplementedError()

        def on_receive(self, sender, message_code, payload):
            """Define the action to take when receiving a package.

            Args:
                sender (int): rank of current process.
                message_code (MessageCode): message code
                payload (torch.Tensor): list[torch.Tensor]
            """
            raise NotImplementedError()

        def setup(self):
            """Initialize network connection and necessary setups.

            Note:
                At first, ``self._network.init_network_connection()`` is required to be called.
                Overwrite this method to implement system setup message communication procedure.
            """
            self._network.init_network_connection()

FedLab provides 2 standard communication pattern implementations: synchronous and asynchronous. You can customize process flow by: 1. create a new class inherited from corresponding class in our standard implementations; 2. overwrite the functions in target communication stage.

To sum up, communication strategy can be customized by overwriting as the note below mentioned.

.. note::

    1. ``setup()`` defines the network initialization stage. Can be used in complex system information synchronize.
    2. ``run()`` is the main process of client. User need to define the communication strategy with user. 
    3. ``on_receive(sender_rank, message_code, payload)`` indicate the control flow and information parsing.

Importantly, ServerManager and ClientManager should be defined and used as a pair. The control flow and information agreements should be compatible. FedLab provides standard implementation for typical synchronous and asynchronous, as depicted below.

Synchronous
============

Synchronous communication involves ``ServerSynchronousManager`` and ``ClientPassiveManager``. Communication procedure is shown as follows.

.. image:: ../imgs/fedlab-synchronous.svg
      :align: center
      :class: only-light

.. image:: ../imgs/fedlab-synchronous-dark.svg
  :align: center
  :class: only-dark

Asynchronous
=============

Asynchronous is given by ``ServerAsynchronousManager`` and ``ClientActiveManager``. Communication
procedure is shown as follows.

.. image:: ../imgs/fedlab-asynchronous.svg
      :align: center
      :class: only-light

.. image:: ../imgs/fedlab-asynchronous-dark.svg
  :align: center
  :class: only-dark

Initialization stage
=======================

Initialization stage is represented by ``manager.setup()`` function.

User can customize initialization procedure as follows(use ClientManager as example):

.. code-block:: python

    from fedlab.core.client.manager import ClientPassiveManager

    class CustomizeClientManager(ClientPassiveManager):

        def __init__(self, handler, network, logger):
            super().__init__(handler, network, logger=logger)

        def setup(self):
            super().setup()
            *****************************
            *                           *
            * Write Customize Code Here *
            *                           *
            *****************************
    
Communication stage
===================

After Initialization Stage, user can define ``run()`` to define main process. To standardilize FedLab's implementation, we encourage users to customize this stage following our code pattern:

.. code-block:: python

    def run(self):
        """Main procedure of each client is defined here:
        1. client waits for data from server （PASSIVE）
        2. after receiving data, client will train local model
        3. client will synchronize with server actively
        """
        self._LOGGER.info("connecting with server")
        self.setup()

        while True: 
            self._LOGGER.info("Waiting for server...")
            # waits for data from server (default server rank is 0)
            sender_rank, message_code, payload = PackageProcessor.recv_package(
                src=0)
            # exit
            if message_code == MessageCode.Exit:
                self._LOGGER.info(
                    "Receive {}, Process exiting".format(message_code))
                self._network.close_network_connection()
                break
            else:
                # perform activation strategy
                self.on_receive(sender_rank, message_code, payload)

            # synchronize with server
            self.synchronize()

Then, put the branch in ``on_receive(sender_rank, message_code, payload)`` function, like this:

.. code-block:: python

    def on_receive(self, sender_rank, message_code, payload):
        """Actions to perform when receiving new message, including local training

        Note:
            Customize the control flow of client corresponding with :class:`MessageCode`.

        Args:
            sender_rank (int): Rank of sender
            message_code (MessageCode): Agreements code defined in :class:`MessageCode`
            payload (list[torch.Tensor]): A list of tensors received from sender.
        """
        self._LOGGER.info("Package received from {}, message code {}".format(
            sender_rank, message_code))
        model_parameters = payload[0]
        self._handler.train(model_parameters=model_parameters)

Shutdown stage
=================

Shutdown stage is started by ServerManager. It will send a package with ``MessageCode.Exit`` to inform ClientManager to stop its process.

.. code-block:: python

    def shutdown_clients(self):
        """Shut down all clients.

        Send package to every client with :attr:`MessageCode.Exit` to ask client to exit.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.

        """
        for rank in range(1, self._network.world_size):
            print("stopping clients rank:", rank)
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=rank)

Example
===========

In fact, the scale module of FedLab is a communication strategy re-definition to both ClientManager and ServerManager. Please see the source code in fedlab/core/{client or server}/scale/__init__.py (It it really simple. We did nothing but add a map function from rank to client id).

