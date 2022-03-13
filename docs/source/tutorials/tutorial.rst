.. _tutorial:

*********
Tutorials
*********

**FedLab** standardizes FL simulation procedure, including synchronous algorithm, asynchronous algorithm :cite:p:`xie2019asynchronous` and communication compression :cite:p:`lin2017deep`. **FedLab** provides modular tools and standard implementations to simplify FL research.

.. toctree::
   :hidden:

   distributed_communication
   communication_strategy
   federated_optimization
   dataset_partition
   docker_deployment


.. card:: Learn Distributed Network Basics
    :link: distributed-communication
    :link-type: ref
    :class-card: sd-rounded-2 sd-border-1

    Step-by-step guide on distributed network setup and package transmission.

.. card:: How to Customize Communication Strategy?
    :link: communication-strategy
    :link-type: ref
    :class-card: sd-rounded-2 sd-border-1

    Use :class:`NetworkManager` to customize communication
    strategies, including synchronous and asynchronous communication.

.. card:: How to Customize Federated Optimization?
    :link: federated-optimization
    :link-type: ref
    :class-card: sd-rounded-2 sd-border-1

    Define your own model optimization process for both server and client.


.. card:: Federated Datasets and Data Partitioner
    :link: dataset-partition
    :link-type: ref
    :class-card: sd-rounded-2 sd-border-1

    Get federated datasets and data partition for IID and non-IID setting.
