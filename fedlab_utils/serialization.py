import torch


class SerializationTool(object):
    @staticmethod
    def serialize_model(model: torch.nn.Module) -> torch.Tensor:
        """Unfold model parameters
        
            
        Unfold every layer of model, concate all of tensors into one.
        Return a `torch.Tensor` with shape (size, )

        Args:
            model (`torch.nn.Module`): model to serialize.
        """

        parameters = [param.data.view(-1) for param in model.parameters()]
        m_parameters = torch.cat(parameters)
        m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def deserialize_model(model: torch.nn.Module,
                          serialized_parameters: torch.Tensor):
        """Assigns serialized parameters to model.parameters.
        This is done by iterating through `model.parameters()` and assigning the relevant params in `grad_update`.
        NOTE: this function manipulates `model.parameters`.

        Args:
            model (`torch.nn.Module`): model to deserialize.
            serialized_parameters (`torch.Tensor`): serialized model parameters.
        """
        
        current_index = 0  # keep track of where to read from grad_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            parameter.data.copy_(
                serialized_parameters[current_index:current_index +
                                      numel].view(size))
            current_index += numel
