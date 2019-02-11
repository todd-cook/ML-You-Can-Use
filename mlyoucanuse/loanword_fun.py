"""`loanword_fun.py` - some functions for loanword processing."""
from typing import Tuple, List, Any

import numpy as np

__author__ = 'Todd Cook <todd.g.cook@gmail.com>'
__license__ = 'MIT License'


def run_length_encoding(in_array: List[int]) -> Tuple[Any, Any, Any]:
    """
    Run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    :param inarray:
    :return: tuple (starts, lengths, values)

    >>> run_length_encoding([0, 0, 0, 1, 1, 1, 0, 0, 0])
    (array([0, 3, 6]), array([3, 3, 3]), array([0, 1, 0]))

    >>> run_length_encoding([0, 0, 0, 0, 0, 0, 0, 0, 0])
    (array([0]), array([9]), array([0]))

    >>> run_length_encoding([1, 1, 1, 1, 1, 1, 1, 1, 1])
    (array([0]), array([9]), array([1]))
    >>> run_length_encoding([0])
    (array([0]), array([1]), array([0]))
    """
    where = np.flatnonzero
    arr = np.asarray(in_array)
    size = len(arr)
    if not size:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=arr.dtype))
    starts = np.r_[0, where(~np.isclose(arr[1:], arr[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, size])
    values = arr[starts]
    return starts, lengths, values


def extract_words(sentence, starts, lengths, values):
    """
    Extract words based on consecutive marks; isolate marks not harvested
    :param sentence:
    :param starts:
    :param lengths:
    :param values:
    :return:

    >>> sent = ['You' , 'know', 'they' ,'say', 'vaya', 'con', 'Dios', 'everytime']
    >>> data = [0, 0, 0, 0, 1, 1, 1, 0]
    >>> extract_words(sent, *run_length_encoding(data))
    [['vaya', 'con', 'Dios']]
    >>> data = [0, 0, 0, 0, 0, 1, 0, 0]
    >>> extract_words(sent, *run_length_encoding(data))
    []

    """
    return [sentence[starts[idx]: starts[idx] + lengths[idx]]
            for idx, tmp in enumerate(values)
            if tmp == 1
            if lengths[idx] > 1
            ]


def extract_consecutive_indices(starts, lengths, values):
    """
    :param starts:
    :param lengths:
    :param values:
    :return:

    >>> sent = ['You' , 'know', 'they' ,'say', 'vaya', 'con', 'Dios', 'everytime']
    >>> data = [0, 0, 0, 0, 1, 1, 1, 0]
    >>> run_length_encoding(data)
    (array([0, 4, 7]), array([4, 3, 1]), array([0, 1, 0]))
    >>> extract_consecutive_indices(*run_length_encoding(data))
    [4, 5, 6]
    >>> sent ='kai to kalon they say often kai to agathon'.split()
    >>> indices = [1,1,1,0,0,0,1,1,1]

    # handle multiple
    >>> extract_consecutive_indices(*run_length_encoding(indices))
    [0, 1, 2, 6, 7, 8]

    """
    places = [(starts[idx], starts[idx] + lengths[idx])
              for idx, tmp in enumerate(values)
              if tmp == 1 and lengths[idx] > 1
              ]
    input_list = [list(range(a, b)) for a, b in places]
    return np.concatenate(input_list).ravel().tolist()
