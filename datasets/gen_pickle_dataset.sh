#!/bin/bash

# Example: bash gen_pickle_dataset.sh "shakespeare" "../datasets" "./pickle_datasets"
# Example: bash gen_pickle_dataset.sh "sent140" "../datasets" "./pickle_datasets" 1
dataset=$1
data_root=${2:-'../datasets'}
pickle_root=${3:-'./pickle_datasets'}
# for nlp datasets
build_vocab=${4:-'0'}
vocab_save_root=${5:-'./nlp_utils/dataset_vocab'}
vector_save_root=${6:-'./nlp_utils/glove'}
vocab_limit_size=${7:-'50000'}


python pickle_dataset.py \
--dataset ${dataset} \
--data_root ${data_root} \
--pickle_root ${pickle_root} \
--build_vocab ${build_vocab} \
--vocab_save_root ${vocab_save_root} \
--vector_save_root ${vector_save_root} \
--vocab_limit_size ${vocab_limit_size}
