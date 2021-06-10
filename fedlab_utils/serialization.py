import torch


class SerializationTool(object):
    @staticmethod
    def serialize_model(model):
        """
        vectorize each model parameter
        """
        parameters = [param.data.view(-1) for param in model.parameters()]
        m_parameters = torch.cat(parameters)
        m_parameters = m_parameters.cpu()

        return m_parameters


    # 考虑删除 改用deserialize_model
    # 函数名匹配
    @staticmethod
    def restore_model(model, serialized_parameters):
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

    @staticmethod
    def deserialize_model(model, serialized_parameters):
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

