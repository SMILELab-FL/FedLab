#!/bin/bash

cd ../data/raw_data

if [ ! -f trainingandtestdata.zip ]; then
    echo "downloading trainingandtestdata.zip"
    wget --no-check-certificate http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
else
  echo "using existing trainingandtestdata.zip"
fi

if [ ! -f training.csv ] || [ ! -f test.csv ]; then
  echo "unzipping trainingandtestdata.zip"
  unzip trainingandtestdata.zip
  mv training.1600000.processed.noemoticon.csv training.csv
  mv testdata.manual.2009.06.14.csv test.csv
else
  echo "using existing training.csv and test.csv in data/raw_data"
fi
#rm trainingandtestdata.zip

cd ../../preprocess