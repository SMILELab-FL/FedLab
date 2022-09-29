#!/bin/bash

if [ ! -d "../data" ]; then
  mkdir ../data
fi
if [ ! -d "../data/raw_data" ]; then
  mkdir ../data/raw_data
fi
if [ ! -f ../data/raw_data/training.csv ] || [ ! -f ../data/raw_data/test.csv ]; then
  echo "------------------------------"
  echo "retrieving raw data"
  
  bash get_data.sh
  echo "finished retrieving raw data"
else
  echo "using existing retrieved raw data"
fi

echo "generating intermediate data"
if [ ! -d "../data/intermediate" ]; then
  mkdir ../data/intermediate
fi

if [ ! "$(ls -A ../data/intermediate)" ]; then
  echo "------------------------------"
  echo "combining raw_data .csv files"
  python3 combine_data.py
  echo "finished combining raw_data .csv files"
else
  echo "using existing retrieved raw data"
fi

if [ ! -d "../data/all_data" ]; then
  mkdir ../data/all_data
fi

if [ ! "$(ls -A ../data/all_data)" ]; then
  echo "------------------------------"
  echo "converting data to .json format"
  python3 data_to_json.py
  echo "finished converting data to .json format"
else
  echo "using existing data/all_data"
fi
