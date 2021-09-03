import sys
import pickle

sys.path.append('../../../../../../')

from fedlab_benchmarks.datasets.leaf_data_process.dataloader import get_LEAF_dataset
from torch.utils.data import ConcatDataset

for i in range(660):
    train, test = get_LEAF_dataset("shakespeare", i)
    file_name = "client" + str(i) + ".pkl"
    with open("./pkl_dataset/test/" + file_name, 'wb') as f:
        pickle.dump(test, f)
    with open("./pkl_dataset/train/" + file_name, 'wb') as f:
        pickle.dump(train, f)
