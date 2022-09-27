#!/bin/bash

if [ ! -d "../data" ]; then
    mkdir ../data
fi

if [ ! -d "../data/raw_data" ]; then
    mkdir ../data/raw_data
fi

if [ ! -f ../data/raw_data/raw_data.txt ]; then
    bash get_data.sh
else
    echo "using existing data/raw_data/raw_data.txt"
fi

if [ ! -d "../data/raw_data/by_play_and_character" ] || [ ! -f ../data/raw_data/users_and_plays.json ]; then
    echo "dividing txt data between users"
    python3 preprocess_shakespeare.py ../data/raw_data/raw_data.txt ../data/raw_data/
else
    echo "using existing divided files of txt data"
fi

RAWTAG=""
if [[ $@ = *"--raw"* ]]; then
  RAWTAG="--raw"
fi
if [ ! -d "../data/all_data" ]; then
    mkdir ../data/all_data
fi
if [ ! "$(ls -A ../data/all_data)" ]; then
    echo "generating all_data.json in data/all_data"
    python3 gen_all_data.py $RAWTAG
fi