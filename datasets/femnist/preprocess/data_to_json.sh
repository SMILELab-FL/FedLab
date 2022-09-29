#!/bin/bash

# assumes that the script is run in the preprocess folder

if [ ! -d "../data" ]; then
  mkdir ../data
fi
if [ ! -d "../data/raw_data" ]; then
  mkdir ../data/raw_data
fi

# download and unzip
if [ ! -d "../data/raw_data/by_class" ] || [ ! -d "../data/raw_data/by_write" ]; then
  echo "------------------------------"
  echo "downloading and unzipping raw data"
  bash get_data.sh
  echo "finished downloading and unzipping raw data"
else
  echo "using existing unzipped raw data folders (by_class and by_write)"
fi


echo "generating intermediate data"
if [ ! -d "../data/intermediate" ]; then # stores .pkl files during preprocessing
  mkdir ../data/intermediate
fi

if [ ! -f ../data/intermediate/class_file_dirs.pkl ]; then
  echo "------------------------------"
  echo "extracting file directories of images"
  python3 get_file_dirs.py
  echo "finished extracting file directories of images"
else
  echo "using existing data/intermediate/class_file_dirs.pkl"
fi

if [ ! -f ../data/intermediate/class_file_hashes.pkl ]; then
  echo "------------------------------"
  echo "calculating image hashes"
  python3 get_hashes.py
  echo "finished calculating image hashes"
else
  echo "using existing data/intermediate/class_file_hashes.pkl"
fi

if [ ! -f ../data/intermediate/write_with_class.pkl ]; then
  echo "------------------------------"
  echo "assigning class labels to write images"
  python3 match_hashes.py
  echo "finished assigning class labels to write images"
else
  echo "using existing data/intermediate/write_with_class.pkl"
fi

if [ ! -f ../data/intermediate/images_by_writer.pkl ]; then
  echo "------------------------------"
  echo "grouping images by writer"
  python3 group_by_writer.py
  echo "finished grouping images by writer"
else
  echo "using existing data/intermediate/images_by_writer.pkl"
fi

if [ ! -d "../data/all_data" ]; then
  mkdir ../data/all_data
fi
if [ ! "$(ls -A ../data/all_data)" ]; then
  echo "------------------------------"
  echo "converting data to .json format in data/all_data"
  python3 data_to_json.py
  echo "finished converting data to .json format"
fi
