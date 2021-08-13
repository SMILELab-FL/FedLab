#!/bin/bash

NAME="celeba"

cd ../utils

python3 stats.py --name $NAME

cd ../$NAME