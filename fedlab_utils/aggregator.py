import torch

class Aggregators(object):
    """Define the algorithm of parameters aggregation"""
    @staticmethod
    def fedavg_aggregate(serialized_params_list):
        """Fedavg aggregator

        paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): merge all Tensors following FedAvg.

        Returns:
            torch.Tensor
        """
        serialized_parameters = torch.mean(
            torch.stack(serialized_params_list), dim=0)
        return serialized_parameters

    @staticmethod
    def fedasgd_aggregate(server_param, new_param, alpha):
        """Fedasgd aggregator
        
        paper: https://arxiv.org/abs/1903.03934
        """
        serialized_parameters = torch.mul(1 - alpha, server_param) + \
                                        torch.mul(alpha, new_param)
        return serialized_parameters
        