# unfinished

from copy import deepcopy
import random
import torch
import torch.distributed as dist
import math


class SubsetSampler(torch.utils.data.Sampler):
    """Subset of a dataset at specified indices. 
        Similar to torch.utils.data.dataset.Subset, but this is a Sampler used in Dataloader

    Args:
        indices (list): Indices in the whole set selected for subset
        shuffle (bool): shuffle the indices or not.
    """
    def __init__(self, indices: list, shuffle=False) -> None:
        self.indices = indices
        if shuffle is True:
            random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


# untested
# modified from DistributedSampler
class FedDistributedSampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """
    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 drop_last=False):

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size() - 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank() - 1
        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval"
                             " [0, {}]".format(rank, num_replicas - 1))

        self._dataset = dataset

        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(
                self.dataset
        ) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                len(self.dataset) /
                self.num_replicas)  # type: ignore[arg-type]

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples:(self.rank + 1) *
                          self.num_samples]
        assert len(indices) == self.num_samples

        if self.shuffle is True:
            random.shuffle(self.indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples


# codes below are about to be abandoned


class DistributedSampler(torch.utils.data.distributed.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """
    def __init__(self, dataset, rank, num_replicas):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            #num_replicas = dist.get_world_size() - 1
            num_replicas = num_replicas
        """
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank() - 1
        """
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank - 1
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class NonIIDDistributedSampler(torch.utils.data.distributed.Sampler):
    """
    This is a copy of :class:`torch.utils.data.distributed.DistributedSampler` (28 March 2019)
    with the option to turn off adding extra samples to divide the work evenly.
    """
    def __init__(self, dataset, rank, num_replicas, add_extra_samples=True):

        if torch.distributed.get_rank() == 0:
            print("Using non iid distributed sampler!!!")

        self._dataset = dataset

        if torch.distributed.is_available():
            self._num_replicas = torch.distributed.get_world_size() - 1
            self._rank = torch.distributed.get_rank() - 1
        else:
            print("warning: torch.distributed is not available")

        self._rank = rank
        self._num_replicas = num_replicas
        self._add_extra_samples = add_extra_samples
        self._epoch = 0

        if add_extra_samples:
            self._num_samples = int(
                math.ceil(len(self._dataset) * 1.0 / self._num_replicas))
            self._total_size = self._num_samples * self._num_replicas
        else:
            self._total_size = len(self._dataset)
            num_samples = self._total_size // self._num_replicas
            rest = self._total_size - num_samples * self._num_replicas
            if self._rank < rest:
                num_samples += 1
            self._num_samples = num_samples

        temp_class_list = []
        for _ in range(len(set(dataset.targets))):
            temp_class_list.append([])

        for index, item_class in enumerate(dataset.targets):
            temp_class_list[item_class].append(index)

        self._indices = []
        for l in temp_class_list:
            self._indices += l

        if self._add_extra_samples:
            self._indices += self._indices[:(self._total_size -
                                             len(self._indices))]
        assert len(self._indices) == self._total_size

    def __iter__(self):
        indices = deepcopy(
            self._indices[self._num_samples * self._rank:self._num_samples *
                          (self._rank + 1)])

        #random.seed(self._epoch)
        #random.shuffle(indices)

        assert len(indices) == self._num_samples

        #self.set_epoch(self._epoch + 1)

        return iter(indices)

    def __len__(self):
        return self._num_samples

    def set_epoch(self, epoch):
        self._epoch = epoch