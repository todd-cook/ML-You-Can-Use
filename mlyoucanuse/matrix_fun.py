"""`matrix_fun.py` - some functions for loanword processing."""
from typing import Tuple, List, Any
import logging
import numpy as np

__author__ = "Todd Cook <todd.g.cook@gmail.com>"
__license__ = "MIT License"

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


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
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=arr.dtype),
        )
    starts = np.r_[0, where(~np.isclose(arr[1:], arr[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, size])
    values = arr[starts]
    return starts, lengths, values


def extract_words(
    sentence: List[str], starts: List[int], lengths: List[int], values: List[int]
):
    """
    Extract words based on consecutive marks; isolate marks not harvested.
    Note, this will return groups of words in separate lists, you will probably want to also use
    merge_words().

    :param sentence:
    :param starts:
    :param lengths:
    :param values:
    :return:

    >>> sent = ['You', 'know', 'they', 'say', 'vaya', 'con', 'Dios', 'everytime']
    >>> data = [0, 0, 0, 0, 1, 1, 1, 0]
    >>> extract_words(sent, *run_length_encoding(data))
    [['vaya', 'con', 'Dios']]

    >>> data = [0, 0, 0, 0, 0, 1, 0, 0]
    >>> extract_words(sent, *run_length_encoding(data))
    []

    """
    return [
        sentence[starts[idx] : starts[idx] + lengths[idx]]
        for idx, tmp in enumerate(values)
        if tmp == 1
        if lengths[idx] > 1
    ]


def merge_words(word_matrix: List[List[str]]) -> List[str]:
    """
    Merge lists of lists into one list.
    :param word_matrix:
    :return:

    >>> merge_words([['kai', 'gar', 'onar'], ['ek', 'Dios', 'esti']])
    ['kai', 'gar', 'onar', 'ek', 'Dios', 'esti']

    """
    if not word_matrix:
        return word_matrix  # type: ignore
    return [item for items in word_matrix for item in items]


def extract_consecutive_indices(
    starts: List[int], lengths: List[int], values: List[int]
):
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

    # handle multiple
    >>> sent ='kai to kalon they say often kai to agathon'.split()
    >>> indices = [1,1,1,0,0,0,1,1,1]
    >>> extract_consecutive_indices(*run_length_encoding(indices))
    [0, 1, 2, 6, 7, 8]

    """
    places = [
        (starts[idx], starts[idx] + lengths[idx])
        for idx, tmp in enumerate(values)
        if tmp == 1 and lengths[idx] > 1
    ]
    input_list = [list(range(a, b)) for a, b in places]
    return np.concatenate(input_list).ravel().tolist()


def match_sequence(arr: List[int], seq: List[int]):
    """
    Given an array, find matching sequences.

    :param arr: List of Integers
    :param seq: List of Integers, typically a smaller sequence
    :return:

    >>> match_sequence([0, 0, 0, 1, 0, 1, 0, 0], [1, 0, 1])
    [3, 4, 5]

    """
    the_arr = np.array(arr)
    the_seq = np.array(seq)
    # Store sizes of input array and sequence
    num_arr, num_seq = the_arr.size, the_seq.size
    # Range of sequence
    r_seq = np.arange(num_seq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    m_vals = (
        the_arr[np.arange(num_arr - num_seq + 1)[:, None] + r_seq] == the_seq
    ).all(1)

    # Get the range of those indices as final output
    if m_vals.any() > 0:
        return np.where(np.convolve(m_vals, np.ones((num_seq), dtype=int)) > 0)[
            0
        ].tolist()
    return []


def patch_cluster_holes(arr: List[int]) -> List[int]:
    """
    Given an array of binary values, heal any holes matching [1, 0, 1]

    :param arr:
    :return:

    >>> patch_cluster_holes([0, 0, 0, 1, 0, 1, 0, 0] )
    [0, 0, 0, 1, 1, 1, 0, 0]

    """
    patches = match_sequence(arr, [1, 0, 1])
    for idx in patches:
        arr[idx] = 1
    return arr
