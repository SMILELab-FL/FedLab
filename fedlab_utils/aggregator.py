# unfinished

import torch


class Aggregators(object):

    @staticmethod
    def fedavg_aggregate(serialized_params_list):
        serialized_parameters = torch.mean(
            torch.stack(serialized_params_list), dim=0)
        return serialized_parameters
