# Datasets Prepare Procedure in FedLab

This folder contains download and preprocessed scripts for commonly used datasets, and provides leaf dataset interface. Each individual folder named by the dataset contains download and preprocess scripts for commonly used datasets.  For LEAF dataset, it contain `celeba`, `femnist`, `reddit`, `sent140`, `shakespeare`, `synthetic`, whose download and preprocess scripts are copied by [LEAF-Github](https://github.com/TalwalkarLab/leaf). For leaf dataset folders, run  `create_datasets_and_save.sh` to get partitioned data. Also we can edit preprocess.sh command params to get a different partition way.


## Process FEMNIST, ShakeSpeare, Sent140, and CelebA

**Set the parameters and instantiate the PickleDataset object (located in pickle_dataset.py), the usage example is as follows:**

```python
from .pickle_dataset import PickleDataset
pdataset = PickleDataset(pickle_root="pickle_datasets", dataset_name="shakespeare")
# create responding dataset in pickle file form
pdataset.create_pickle_dataset(data_root="../datasets")
# read saved pickle dataset and get responding dataset
train_dataset = pdataset.get_dataset_pickle(dataset_type="train", client_id="0")
test_dataset = pdataset.get_dataset_pickle(dataset_type="test", client_id="2")
```

Parameter Description:

1. `data_root`: The root path for storing leaf data sets, which contains leaf data sets; if you use the `fedlab_benchmarks/datasets/` provided by fedlab to download leaf data, then `data_root` can be set to this path. The relative address of the path is out.
2. `pickle_root`: Store the pickle file address of the processed DataSet, each data set _DataSet_ will be saved as `{pickle_root}/{dataset_name}/{train,test}`; the example is to create a `pickle_datasets` folder under the current path Store all pickle dataset files.
3. `dataset_name`: Specify the name of the leaf data set to be processed. There are six options {feminist, Shakespeare, celeba, sent140, synthetic, reddit}.

> Besides, you can directly run the `gen_pickle_dataset.sh` script (located in `fedlab_benchmarks/leaf`) to instantiate the corresponding PickleDataset object for the dataset and store it as a pickle file.
```shell
bash gen_pickle_dataset.sh "femnist" "../datasets" "./pickle_datasets"
```
And parameters 1, 2, and 3 correspond to dataset_name, data_root, and pickle_root respectively.


## References

- [Caldas, Sebastian, et al. "Leaf: A benchmark for federated settings." arXiv preprint arXiv:1812.01097 (2018).](https://arxiv.org/abs/1812.01097)
- [Li, Qinbin, et al. "Federated learning on non-iid data silos: An experimental study." 2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE, 2022.](https://ieeexplore.ieee.org/abstract/document/9835537/)