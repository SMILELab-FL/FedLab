# LEAF_DATA README

This folder contains LEAF data download and preprocessed scripts, which are copied 
    by [LEAF-Github](https://github.com/TalwalkarLab/leaf)

-  LEAF datasets contain `celeba`, `femnist`, `reddit`, `sent140`, `shakespeare`, `synthetic`, 
-  Each subfolder is named by dataset name, like `femnist`: contains preprocess scripts. 
   Run  `run.sh` to get partitioned data as an example for each dataset.
-  For all datasets, the preprocess should split data into train and test. 
    After generating `{dataset}/data/train` and `{dataset}/data/test`, the data in `{dataset}/data` are useless.
    Users can delete the original data as needed.
