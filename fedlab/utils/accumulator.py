import torch

from copy import deepcopy

from torch.autograd import grad


class GradientAccumulator(object):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = deepcopy(model)

        self.gradients = [torch.zeros_like(parameter.data) for parameter in self.model.parameters()]

    def reset(self):
        # reset all variables
        self.gradients = [torch.zeros_like(parameter) for parameter in self.gradients]

    def delta(self, model):
        delta = []
        for param_t, param_t0 in zip(model.parameters(), self.model.parameters()):
            delta_ = param_t.data.detach() - param_t0.data.detach()
            delta.append(delta_)

    def accumulate(self, model):
        for index, parameters in enumerate(model.parameters()):
            self.gradients[index].add(parameters.grad.detach())

    @property
    def gradients(self):
        return torch.cat(self.gradients)