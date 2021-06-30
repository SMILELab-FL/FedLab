import torch

class Aggregators(object):
    """Define the algorithm of parameters aggregation"""
    @staticmethod
    def fedavg_aggregate(serialized_params_list):
        serialized_parameters = torch.mean(
            torch.stack(serialized_params_list), dim=0)
        print("Aggregators.fedavg_aggregate:  merged shape ",serialized_parameters.shape)
        return serialized_parameters

    @staticmethod
    def fedasgd_aggregate(server_param, new_param, alpha):
        serialized_parameters = torch.mul(1 - alpha, server_param) + \
                                        torch.mul(alpha, new_param)
        return serialized_parameters
        