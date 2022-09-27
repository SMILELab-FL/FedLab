# Datasets Prepare Procedure in FedLab

This folder contains download and preprocessed scripts for commonly used datasets, and provides leaf dataset interface. Each individual folder named by the dataset contains download and preprocess scripts for commonly used datasets.  For LEAF dataset, it contain `celeba`, `femnist`, `reddit`, `sent140`, `shakespeare`, `synthetic`, whose download and preprocess scripts are copied by [LEAF-Github](https://github.com/TalwalkarLab/leaf). For leaf dataset folders, run  `create_datasets_and_save.sh` to get partitioned data. Also we can edit preprocess.sh command params to get a different partition way.

## References

- [Caldas, Sebastian, et al. "Leaf: A benchmark for federated settings." arXiv preprint arXiv:1812.01097 (2018).](https://arxiv.org/abs/1812.01097)
- [Li, Qinbin, et al. "Federated learning on non-iid data silos: An experimental study." 2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE, 2022.](https://ieeexplore.ieee.org/abstract/document/9835537/)