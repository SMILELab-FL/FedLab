# LEAF_DATA_PROCESS README

This folder contains process method to read LEAF data and get dataloader process for users.

-  `json_data_read_util.py`: read data for leaf_data_process processed json files.
    This is modified by [LEAF/models/utils/model_utils.py](https://github.com/TalwalkarLab/leaf/blob/master/models/utils/model_utils.py)
-  `dataloader.py`: provide methods to get data and dataloader for dataset in LEAF.
-  `dataset` folder: provide each dataset's `torch.utils.data.Dataset` Dataset