# DATASETS README

This folder contains download and preprocessed scripts for commonly used datasets, and provides leaf dataset interface.


- `data` folder: contains download and preprocess scripts for commonly used datasets, each subfolder is named by dataset name. 

  For LEAF dataset, it contain `celeba`, `femnist`, `reddit`, `sent140`, `shakespeare`, `synthetic`, whose download and preprocess scripts are copied by [LEAF-Github](https://github.com/TalwalkarLab/leaf). And we copy and modify from [Flower leaf script](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/scripts/leaf/femnist]) to store processed `Dataset` in the form of pickle files.

  For leaf dataset folders, run  `create_datasets_and_save.sh` to get partitioned data. Also we can edit preprocess.sh command params to get a different partition way.

- `leaf_data_process` folder: contains process method to read leaf data and get dataloader for users. 