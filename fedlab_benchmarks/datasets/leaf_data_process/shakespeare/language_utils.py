"""
    Utils for shakespeare dataset
    This is copied by [LEAF](https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py)
    Vocabulary re-used from the Federated Learning for Text Generation tutorial.
    https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
"""

CHAR_VOCAB = list(
    'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'
)

# ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
ALL_LETTERS = "".join(CHAR_VOCAB)

# Vocabulary with OOV ID, zero for the padding, and BOS, EOS IDs.
VOCAB_SIZE = len(ALL_LETTERS) + 4


def _one_hot(index, size):
    """Returns one-hot vector with given size and value 1 at given index

    Args:
        index: value 1 at index
        size: length of one-hot vector

    Returns: one-hot vector
    """
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    """Returns one-hot index representation of given letter in ALL_LETTERS

    Args:
        letter (char/str[0]): input letter

    Returns: one-hot vector for input letter with size = VOCAB_SIZE

    """
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, VOCAB_SIZE)


def letter_to_index(letter):
    """Returns index in ALL_LETTERS of given letter

    Args:
        letter (char/str[0]): input letter

    Returns: int index of input letter
    """
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):
    """Returns a list of character indices

    Args:
        word (str): input string

    Return:
        indices: int list with length len(word)
    """
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices