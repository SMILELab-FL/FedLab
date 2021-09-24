.. _data-partition:

***************
DataPartitioner
***************

Complicated in real world, FL need to handle various kind of data distribution scenarios, including
iid and non-iid scenarios. Though there already exists some partition schemes for published data benchmark,
it still can be very messy and hard for researchers to partition datasets according to their specific
research problems, and maintain partition results during simulation. FedLab provides :class:`fedlab.utils.dataset.partition.DataPartition` that allows you to use pre-partitioned datasets as well as your own data. :class:`DataPartitioner` stores sample indices for each client given a data partition scheme.

FedLab provides a number of pre-defined partition schemes for some datasets (such as CIFAR10) that subclass :class:`fedlab.utils.dataset.partition.DataPartition` and implement functions specific to particular partition scheme. They can be used to prototype and benchmark your FL algorithms.


CIFAR10Partitioner
==================

For CIFAR10, we provides 6 pre-defined partition schemes. We partition CIFAR10 with the following parameters:

- ``targets`` is labels of dataset to partition
- ``num_clients`` specifies number of clients in partition scheme
- ``balance`` refers to FL scenario that sample numbers for different clients are the same
- ``partition`` specifies partition scheme name
- ``unbalance_sgm`` is parameter for unbalance partition
- ``num_shards`` is parameter for non-iid partition using shards
- ``dir_alpha`` is parameter for Dirichlet distribution used in partition
- ``verbose`` controls whether to print intermediate information
- ``seed`` sets the random seed

Each partition scheme can be applied on CIFAR10 using different combinations of parameters:

- ``balance=None``: do not specify sample numbers for each clients in advance

  - ``partition="dirichlet"``: non-iid partition used in
    :cite:t:`yurochkin2019bayesian` and :cite:t:`wang2020federated`. ``dir_alpha`` need to be specified in this partition scheme

  - ``partition="shards"``: non-iid method used in FedAvg `paper <https://arxiv.org/abs/1602.05629>`_. Refer to :func:`fedlab.utils.dataset.functional.shards_partition` for more information. ``num_shards`` need to be specified here.

- ``balance=True``: "Balance" refers to FL scenario that sample numbers for different clients are the same. Refer to :func:`fedlab.utils.dataset.functional.balance_partition` for more information. This partition scheme is from :cite:t:`acar2020federated`.

  - ``partition="iid"``: Random select samples from complete dataset given sample number for each client.

  - ``partition="dirichlet"``: Refer to :func:`fedlab.utils.dataset.functional.client_inner_dirichlet_partition` for more information. ``dir_alpha`` need to be specified in this partition scheme

- ``balance=False``: "Unbalance" refers to FL scenario that sample numbers for different clients are different. For unbalance method, sample number for each client is drown from Log-Normal distribution with variance ``unbalanced_sgm``. When ``unbalanced_sgm=0``, partition is balanced. This partition scheme is from :cite:t:`acar2020federated`.

  - ``partition="iid"``: Random select samples from complete dataset given sample number for each client.

  - ``partition="dirichlet"``: Given sample number of each client, use Dirichlet distribution for each client's class distribution. ``dir_alpha`` need to be specified in this partition scheme

To conclude, 6 pre-defined partition schemes can be summarized as:

- Hetero Dirichlet (non-iid)
- Shards (non-iid)
- Balanced IID (iid)
- Unbalanced IID (iid)
- Balanced Dirichlet (non-iid)
- Unbalanced Dirichlet (non-iid)

In codes above, data\_indices is a dictionary (in order to ensure that the partition result is consistent in the case of cross-machine, the user can save the division dictionary in a file) like this：

.. code:: python

    dict= { '0': indices of dataset,
            '1': indices of dataset,
            ...
            'k': indices of dataset }

By using torch's sampler, only the right part of the sample is taken from the overall dataset.

.. code:: python

    from fedlab.utils.dataset.sampler import SubsetSampler

    train_loader = torch.utils.data.DataLoader(
                    trainset,
                    sampler=SubsetSampler(indices=data_slices[client_id],
                                          shuffle=True),
                    batch_size=batch_size)

There is also a similar implementation of directly reordering and partition the dataset, see fedlab.utils.dataset.sampler.RawPartitionSampler for details.

In addition to dividing the dataset by the sampler of torch, dataset can also be divided directly by splitting the dataset file. The implementation can refer to FedLab version of LEAF.
