import torch

"""
压缩模型参数
"""


def ravel_model_params(model, grads=False, cuda=False):
    """
    Squash model parameters or gradients into a single tensor.
    """
    if cuda:
        m_parameter = torch.Tensor([0]).cuda()
    else:
        m_parameter = torch.Tensor([0])
    for parameter in list(model.parameters()):
        if grads:
            m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
        else:
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
    return m_parameter[1:]


def unravel_model_params(model, parameter_update):
    """
    Assigns grad_update params to model.parameters.
    This is done by iterating through model.parameters() and assigning the relevant params in grad_update.
    NOTE: this function manipulates model.parameters.
    """
    current_index = 0  # keep track of where to read from grad_update
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        parameter.data.copy_(parameter_update[current_index:current_index + numel].view(size))
        current_index += numel


def unravel_model_grad(model, grad_update):
    """
    Assigns grad_update params to model.parameters.
    This is done by iterating through model.parameters() and assigning the relevant params in grad_update.
    NOTE: this function manipulates model.parameters.
    """
    current_index = 0  # keep track of where to read from grad_update
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        parameter.grad.copy_(grad_update[current_index:current_index + numel].view(size))
        current_index += numel