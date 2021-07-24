# DATASETS README

This folder contains leaf data download and preprocessed script, and process code for users


-  `leaf_data` folder: contains LEAF datasets download and preprocess scripts, which are copied 
    by [LEAF-Github](https://github.com/TalwalkarLab/leaf)
    - LEAF datasets contain `celeba`, `femnist`, `reddit`, `sent140`, `shakespeare`, `synthetic`.
    - Each subfolder is named by dataset name. Run  `run.sh` to get partitioned data as an example for each dataset.
    - For all datasets, the preprocess should split data into train and test. 
        After generating `{dataset}/data/train` and `{dataset}/data/test`, the data in `{dataset}/data` are useless,
        Users can delete the original data as needed.
-  `leaf_data_process` folder: contains process method to read leaf data and get dataloader for users. 
    
