from .base_sampler import FedSampler
import numpy as np

class RandomSampler(FedSampler):
    def __init__(self, n, probs=None):
        self.name = "random_sampling"
        self.n = n
        self.p = probs if probs is not None else np.ones(n) / float(n)

        assert len(self.p) == self.n

    def sample(self, k, replace=False):
        if k == self.n:
            self.last_sampled = np.arange(self.n), self.p
            return np.arange(self.n)
        else:
            sampled = np.random.choice(self.n, k, p=self.p, replace=replace)
            self.last_sampled = sampled, self.p[sampled]
            return np.sort(sampled)

    def update(self, probs):
        self.p = probs
