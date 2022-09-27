#!/bin/bash

# download data and convert to .json format

RAWTAG=""
if [[ $@ = *"--raw"* ]]; then
  RAWTAG="--raw"
fi

if [ ! -d "data/all_data" ] || [ ! "$(ls -A data/all_data)" ]; then
    cd preprocess
    bash data_to_json.sh $RAWTAG
    cd ..
else
    echo "using existing data/all_data data folder to preprocess"
fi

NAME="shakespeare"

cd ../utils

bash preprocess.sh --name $NAME $@

cd ../$NAME