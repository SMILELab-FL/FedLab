from abc import ABCMeta, abstractmethod

class FedSampler:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, n):
        self.n = n
    
    @abstractmethod
    def candidate(self, size):
        pass

    @abstractmethod
    def sample(self, size):
        pass

    @abstractmethod
    def update(self, val):
        pass
