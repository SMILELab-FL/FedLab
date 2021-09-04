#!/bin/bash
# copy and modify from Flower leaf script
# [https://github.com/adap/flower/tree/main/baselines/flwr_baselines/scripts/leaf/femnist]

# Delete previous partitions if needed then extract the entire dataset
echo 'Deleting previous dataset split.'
if [ -d "data" ]; then rm -rf "data" ; fi
if [ -d "meta" ]; then rm -rf "meta" ; fi

echo 'Creating new LEAF dataset split.'
bash preprocess.sh -s niid --sf 1 -k 0 -t sample --tf 0.9

# Save train/test dataset in pickle
python preprocess/pickle_dataset.py \
--save_root "data/pickle_dataset" \
--leaf_train_jsons_root "data/train" \
--leaf_test_jsons_root "data/test"
echo 'Done'