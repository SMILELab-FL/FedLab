# NLP——UTILS

This folder contains some lightweight utils for nlp process.

- get_glove.sh: glove download script, from http://nlp.stanford.edu/data/glove.6B.zip to get glove.6B.300d.txt.

- build_vocab.sh: provide a way to sample some clients' train data for building a vocabulary for federated nlp tasks, 
which is a simple alternative method compared with use all data directly in other implementation.

- sample_build_vocab.py: provide a way to sample some clients' train data for building a vocabulary for federated nlp tasks. 

- tokenizer.py: provide `class Tokenizer`, splitting an entire text into smaller units called tokens, such as individual words or terms.
    
- vocab.py: provide `class Vocab`, to encapsulate vocabulary operations in nlp, 
such as getting word2idx for tokenized input data, and get vector list from pretrained word_vec_file, such as glove.6B.300d.txt.