# This is modified by [LEAF/models/sent140/get_embs.sh]
# https://github.com/TalwalkarLab/leaf/blob/master/models/sent140/get_embs.sh

#!/usr/bin/env bash

if [ ! -f 'glove.6B.300d.txt' ]; then
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    rm glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt glove.6B.zip
fi