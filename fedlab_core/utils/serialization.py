import torch

def ravel_model_params(model, cuda=False):
    """
    Squash model parameters or gradients into a single tensor.
    """
    parameters = [param.data.view(-1) for param in model.parameters()
                  ]  # vectorize each model parameter
    m_parameters = torch.cat(parameters)

    if cuda:
        m_parameters = m_parameters.cuda()

    return m_parameters


def unravel_model_params(model, serialized_parameters):
    """
    Assigns grad_update params to model.parameters.
    This is done by iterating through `model.parameters()` and assigning the relevant params in `grad_update`.
    NOTE: this function manipulates `model.parameters`.
    """
    current_index = 0  # keep track of where to read from grad_update
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        parameter.data.copy_(
            serialized_parameters[current_index:current_index + numel].view(size))
        current_index += numel
