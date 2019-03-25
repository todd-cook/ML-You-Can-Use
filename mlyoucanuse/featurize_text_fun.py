"""`featurize_text_fun.py` - functions useful for featurizing text."""

import logging
from typing import List, Dict, Any

import numpy as np

__author__ = 'Todd Cook <todd.g.cook@gmail.com>'
__license__ = 'MIT License'

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def max_suffix(word, max_len=9):
    """

    :param word:
    :param max_len:
    :return:

    >>> max_suffix('doctissimorum')
    'issimorum'
    >>> max_suffix('civis')
    'civis'
    >>> max_suffix('amantur')
    'antur'

    """
    if not word:
        return word
    if len(word) > 11:
        return word[- max_len:]
    if len(word) < 6:
        return word
    return word[-(len(word) - 2):]


def featurize(sentence: List[str], idx: int) -> Dict[str, Any]:
    """
    Create a dictionary for features, use a rolling window of 5 words.
    :param sentence: a list of tokenized words
    :param idx: the current desired word position
    :return: a dictionary of param values

    >>> params = featurize(['quick', 'brown', 'fox', 'jumped', 'over'], 2)
    >>> params['ante_previous_word_suffix'] =='quick'
    True
    >>> params['previous_word_suffix'] =='brown'
    True
    >>> params['penultimate_suffix'] == 'mped'
    True
    >>> params['ultimate_suffix'] == 'over'
    True

    """
    if not sentence or idx > len(sentence):
        return {}  # type: ignore
    return {
        'word': sentence[idx],
        'first_position': idx == 0,
        'last_position': idx == len(sentence) - 1,
        'initial_capitalization': sentence[idx][0].upper() == sentence[idx][0],
        'all_caps': sentence[idx].upper() == sentence[idx],
        'all_lower': sentence[idx].lower() == sentence[idx],
        'ante_previous_word_suffix': '' if idx <= 1 else max_suffix(sentence[idx - 2]),
        'previous_word_suffix': '' if idx == 0 else max_suffix(sentence[idx - 1]),
        'word_suffix': max_suffix(sentence[idx]),
        'penultimate_suffix': '' if idx + 1 >= len(sentence) - 1 else max_suffix(sentence[idx + 1]),
        'ultimate_suffix': '' if idx + 2 > len(sentence) - 1 else max_suffix(sentence[idx + 2]),
    }


def word_to_features(word: str, max_word_length: int = 20, reverse: bool = True) -> List[int]:
    """

    :param word: a single word
    :param max_word_length: the maximum word length for the feature array. If the word is longer
    than this, it will be truncated, if reverse is True, the prefix of the word will be trimmed.
    :param reverse: flip the word, to align words by suffixes.
    :return: A list of ordinal integers mapped to each character and padded to the max word length.

    >>> word_to_features('far')
    [114, 97, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> word_to_features('far', 5)
    [114, 97, 102, 0, 0]

    """
    wordlist = list(word)
    if reverse:
        wordlist.reverse()
    if len(wordlist) > max_word_length:
        LOG.warning('Excessive word length %s for %s, truncating to %s', len(word), word,
                    max_word_length)
        wordlist = wordlist[:max_word_length]
    replacer = {32: 0}  # in a feature matrix a space should be a zero, let's force it
    return [replacer.get(ord(c), ord(c)) for c in "".join(wordlist).ljust(max_word_length, ' ')]


def vectorize_features(params):
    """

    :param params:
    :return:
    """

    return np.concatenate((word_to_features(params['word'], max_word_length=21),
                           0 if not params['first_position'] else 1,
                           0 if not params['last_position'] else 1,
                           0 if not params['initial_capitalization'] else 1,
                           0 if not params['all_caps'] else 1,
                           0 if not params['all_lower'] else 1,
                           word_to_features(params['ante_previous_word_suffix'], max_word_length=9),
                           word_to_features(params['previous_word_suffix'], max_word_length=9),
                           word_to_features(params['penultimate_suffix'], max_word_length=9),
                           word_to_features(params['ultimate_suffix'], max_word_length=9)
                           ), axis=None).tolist()
