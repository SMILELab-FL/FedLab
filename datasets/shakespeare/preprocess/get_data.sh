#!/bin/bash

cd ../data/raw_data

if [ ! -f 1994-01-100.zip ]; then
  wget http://www.gutenberg.org/files/100/old/1994-01-100.zip
else
  echo "using existing 1994-01-100.zip"
fi

echo "unzipping 1994-01-100.zip"
unzip 1994-01-100.zip
#rm 1994-01-100.zip
mv 100.txt raw_data.txt

#wget --adjust-extension http://www.gutenberg.org/files/100/100-0.txt
#mv 100-0.txt raw_data.txt

cd ../../preprocess