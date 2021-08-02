"""
    This is copied by [LEAF](https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py)
    Utils for sent140 dataset
    Download `embs.json` by running 'sent140/get_embs.sh' to use
"""
import os
import re
import numpy as np
import json


def get_word_emb_arr(path="embs.json"):
    """Return word to embedding array, vocab dict and index of each word in vocab by reading `embs.json`
    Args:
        path: the directory for `embs.json`
    Returns:
        each word to embedding array, vocab dict and index of each word in vocab
    Examples:
        word_emb_arr, indd, vocab = get_word_emb_arr("embs.json")
    Raises:
        FileNotFoundError: [Errno 2] No such file or directory: `path`
    """
    # get absolute path
    path = os.path.join(os.path.abspath(__file__), path)
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab


def _split_line(line):
    """Split given line/phrase into list of words
    Args:
        line (str): string representing phrase to be split
    Return:
        list of strings, with each string representing a word
    """
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    """Returns index of given word based on given lookup dictionary
    if word is in look dictionary, return index
    else return the length of the lookup dictionary.
    Args:
        word (str): input word string
        indd (dict): dictionary with string words as keys and int indices as values
    Returns:
        index of given word based on given lookup dictionary, or the length of the lookup dictionary.
    """
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    """Converts given phrase into list of word indices
    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer
    representing unknown index to returned list until the list's length is
    max_words
    Args:
        line (str): string representing phrase/sequence of words
        word2id (dict): dictionary with string words as keys and int indices as values
        max_words (int): maximum number of word indices in returned list
    Return:
        indl (list): list of word indices, one index for each word in phrase
    """
    unk_id = len(word2id)
    line_list = _split_line(line)  # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id] * (max_words - len(indl))
    return indl


def bag_of_words(line, vocab):
    """Returns bag of words representation of given phrase using given vocab
    Args:
        line (str): string representing phrase to be parsed
        vocab (optional): dictionary with words as keys and indices as values.
            Defaults to var `VOCAB_DICT`, from `get_word_meb_arr.py`
    Return:
        integer list: number of occurrences of each word in the string using given vocab
    """
    bag = [0] * len(vocab)
    words = _split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def val_to_vec(size, val):
    """Converts target into one-hot.
    Args:
        size (int): Size of vector.
        val (int): Integer in range [0, size].
    Returns:
         vec: one-hot vector with a 1 in the val element.
    """
    assert 0 <= val < size
    vec = [0 for _ in range(size)]
    vec[int(val)] = 1
    return vec