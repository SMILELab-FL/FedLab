# NLP——UTILS

This folder contains some utils for nlp process.

-  get_glove.sh: glove download script, from http://nlp.stanford.edu/data/glove.6B.zip to get glove.6B.300d.txt.

-  tokenizer.py: provide `class Tokenizer`, splitting an entire text into smaller units called tokens, such as individual words or terms.
    This is copy from [RSE-Adversarial-Defense-Github](https://github.com/Raibows/RSE-Adversarial-Defense/tree/de7bb5afc94d3d262cf0b08f55952800161865ce)
    
-  vocab.py: provide `class Vocab`, to encapsulate vocabulary operations in nlp,
    such as getting word2idx for tokenized input data, and get vector list from pretrained word_vec_file, such as glove.6B.300d.txt.
    This is copy from [RSE-Adversarial-Defense-Github](https://github.com/Raibows/RSE-Adversarial-Defense/tree/de7bb5afc94d3d262cf0b08f55952800161865ce)