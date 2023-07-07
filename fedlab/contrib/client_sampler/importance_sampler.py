from .base_sampler import FedSampler
import numpy as np



class MultiArmedBanditSampler(FedSampler):
    "Refer to [Stochastic Optimization with Bandit Sampling](https://arxiv.org/abs/1708.02544)."

    def __init__(self, n, T, L):
        super().__init__(n)
        self.name = "mabs"
        self.w = np.ones(n)
        self.p = np.ones(n) / float(n)

        self.eta = 0.4
        self.delta = np.sqrt(
            (self.eta**4) * np.log(self.n) / ((self.n**5) * T * (L**2)))
        self.last_sampled = None

    def sample(self, batch_size):
        sampled = np.random.choice(np.arange(self.n),
                                   size=batch_size,
                                   replace=True,
                                   p=self.p)
        p = self.p[sampled]
        self.last_sampled = (sampled, p)
        return np.sort(sampled)

    def update(self, loss):
        at = loss**2 / (self.n**2)
        indices, p = self.last_sampled
        self.w[indices] *= np.exp(self.delta * at / p**3)
        self.p = (1 - self.eta) * self.w / np.sum(self.w) + self.eta / self.n


class OptimalSampler(FedSampler):
    "Refer to [Optimal Client Sampling for Federated Learning](arxiv.org/abs/2010.13723)."

    def __init__(self, n, k):
        super().__init__(n)
        self.name = "optimal"
        self.k = k
        self.p = None

    def sample(self, size=None):
        indices = np.arange(
            (self.n))[np.random.random_sample(self.n) <= self.p]
        self.last_sampled = indices, self.p[indices]
        return indices

    def update(self, loss):
        self.p = self.optim_solver(loss)

    def optim_solver(self, norms):
        norms = np.array(norms)
        idx = np.argsort(norms)
        probs = np.zeros(len(norms))
        l = 0
        for l, id in enumerate(idx):
            l = l + 1
            if self.k + l - self.n > sum(norms[idx[0:l]]) / norms[id]:
                l -= 1
                break

        m = sum(norms[idx[0:l]])
        for i in range(len(idx)):
            if i <= l:
                probs[idx[i]] = (self.k + l - self.n) * norms[idx[i]] / m
            else:
                probs[idx[i]] = 1

        return np.array(probs)

    # def estimate(self):
    #     indices = np.arange(
    #         (self.n))[np.random.random_sample(self.n) <= self.p]
    #     return indices, self.p[indices]