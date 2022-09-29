#!/bin/bash

# assumes that the script is run in the preprocess folder

cd ../data/raw_data
if [ ! -f by_class.zip ]; then
  echo "downloading by_class.zip"
  wget https://s3.amazonaws.com/nist-srd/SD19/by_class.zip
else
  echo "using existing by_class.zip"
fi

if [ ! -f by_write.zip ]; then
  echo "downloading by_write.zip"
  wget https://s3.amazonaws.com/nist-srd/SD19/by_write.zip
else
  echo "using existing by_write.zip"
fi

if [ ! -d "by_class" ]; then
  echo "unzipping by_class.zip"
  unzip by_class.zip
  #rm by_class.zip
else
  echo "using existing unzipped folder by_class"
fi

if [ ! -d "by_write" ]; then
  echo "unzipping by_write.zip"
  unzip by_write.zip
  #rm by_write.zip
else
  echo "using existing unzipped folder by_write"
fi

cd ../../preprocess
