"""`text_deduplicater.py` - Document deduplication helper class.

    Methods for shingling documents in a scalable manner.

    The algorithm in this package is currently restricted to 32 bit integer hash codes.
    The code was heavily adapted from:
    readings from Mining Massive Datasets http://www.mmds.org/
    http://mccormickml.com/2015/06/12/minhash-tutorial-with-python-code
"""
import binascii
import logging
import sys
import unicodedata
import random
import re
from collections import defaultdict
from itertools import permutations
from typing import List, Dict, Set, Tuple, Optional

import numpy as np

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

# the maximum shingle hash that we can assign
MAX_SHINGLE_HASH = 2 ** 32 - 1

# We need the next largest prime number above 'MAX_SHINGLE_HASH'.
# I looked this value up here:
# http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
PRIME_ABOVE_MAX_HASH = 4294967311


# pylint: disable=too-many-instance-attributes
class TextDeduplicater:
    """
    Utility class for deduplicating text documents.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        coeff_a: List[int] = None,
        coeff_b: List[int] = None,
        num_hash_fun: int = 10,
        drop_punctuation: bool = True,
        max_hash: int = MAX_SHINGLE_HASH,
        prime_above_max_hash: int = PRIME_ABOVE_MAX_HASH,
    ):
        """

        :param coeff_a: a list of integers to seed the random hash functions
        :param coeff_b: a second list of integers to seed the random hash functions
        :param num_hash_fun: the number of hash functions to use
        :param drop_punctuation: boolean, whether or not to swallow punctuation
        :param max_hash: the max hash value to use for rollover
        :param prime_above_max_hash: the first prime above the given max hash
        """
        self.hash_data = []  # type: List[List[int]]
        self.idx_to_doc_name = {}  # type: Dict[int,str]
        self.curr_idx = 0
        self.num_hash_fun = num_hash_fun
        self.coeff_a = pick_random_coeffs(num_hash_fun) if not coeff_a else coeff_a
        self.coeff_b = pick_random_coeffs(num_hash_fun) if not coeff_b else coeff_b
        self.punctuation_substitutions = (
            punctuation_for_spaces_dict() if drop_punctuation else None
        )
        self.max_hash = max_hash
        self.prime_above_max_hash = prime_above_max_hash
        self.newline_pattern = re.compile(r"\n")
        self.multispace_pattern = re.compile(r"\s+")

    # pylint: disable=too-many-locals

    def add_document(self, doc_name: str, text: str) -> Optional[int]:
        """
        Add a document to our hash matrix. We only retain the filename and num_hash_fun integers.
        :param doc_name:
        :param text:
        :return: the index of the document added or none

        >>> deduper = TextDeduplicater()
        >>> doc = 'The quick brown fox jumped over the lazy dog.'
        >>> deduper.add_document('myfile1', doc)
        0
        >>> deduper.add_document('bad file', 'no trigram')
        >>> doc3 = 'Sphinx of black quartz, judge my vow.'
        >>> deduper.add_document('myfile3', doc3)
        1

        """
        if self.punctuation_substitutions:
            text = text.translate(self.punctuation_substitutions)
        # Remove newlines and extra spaces
        text = self.newline_pattern.sub(" ", text)
        text = self.multispace_pattern.sub(" ", text)
        word_list = text.split(" ")
        trigrams = grammify(word_list=word_list, num=3)
        # Hash the shingle to a 32-bit integer.
        shingles = []  # type: List[int]
        if not trigrams:
            LOG.warning("No trigrams found for %s", doc_name)
            return None

        # Hurray, we have trigrams to process; increment counter and record doc name
        doc_idx = self.curr_idx
        self.idx_to_doc_name[doc_idx] = doc_name
        self.curr_idx += 1

        for shingle in trigrams:
            shingle_string = " ".join(list(shingle))
            crc = binascii.crc32(shingle_string.encode(encoding="UTF-8")) & 0xFFFFFFFF
            shingles.append(crc)
        # The resulting minhash signature for this document.
        signature = []
        # For each of the random hash functions...
        for idx in range(0, self.num_hash_fun):
            # For each of the shingles actually in the document, calculate its hash code
            # using hash functions[idx]. Track the lowest hash ID seen.
            # Initialize 'minHashCode' to be greater than the maximum possible value output
            #  by the hash.
            min_hash_code = self.prime_above_max_hash + 1
            # For each shingle in the document...
            for shingle_id in shingles:
                # Evaluate the hash function.
                hash_code = (
                    self.coeff_a[idx] * shingle_id + self.coeff_b[idx]
                ) % self.prime_above_max_hash
                # Track the lowest hash code seen.
                if hash_code < min_hash_code:
                    min_hash_code = hash_code
            # Add the smallest hash code value as component number idx of the signature.
            signature.append(min_hash_code)
        self.hash_data.append(signature)
        return doc_idx

    def get_unique_doc_names(self) -> List[str]:
        """
        Get names of unique documents
        :return:

        >>> deduper = TextDeduplicater()
        >>> doc = 'The quick brown fox jumped over the lazy dog.'
        >>> deduper.add_document('myfile1', doc)
        0
        >>> deduper.add_document('bad file', 'no trigram')
        >>> doc3 = 'Sphinx of black quartz, judge my vow.'
        >>> deduper.add_document('myfile3', doc3)
        1
        >>> deduper.add_document('myfile2', doc)
        2
        >>> sorted(deduper.get_unique_doc_names())
        ['myfile1', 'myfile3']

        """
        new_arr = [tuple(row) for row in self.hash_data]  # type: ignore
        _, indices = np.unique(new_arr, axis=0, return_index=True)
        names = [self.idx_to_doc_name[idx] for idx in indices]  # type: List[str]
        return names

    # pylint: disable=too-many-locals
    def get_possible_duplicate_doc_names(
        self, threshold: float = 1.0
    ) -> List[Tuple[str, str]]:
        """
        Get list of duplicate document names.

        We reduce our search space from N**2 to M**2 where M is the number of rows where at least
        one column matches with another one.

        :param threshold: the percentage of matching columns, 1.0 for exact match. 0.8 for 80%, etc.
        :return: a list of tuples of possible duplicate document name pairs

        >>> deduper = TextDeduplicater()
        >>> doc = 'The quick brown fox jumped over the lazy dog.'
        >>> deduper.add_document('myfile1', doc)
        0
        >>> deduper.add_document('bad file', 'no trigram')
        >>> doc3 = 'Sphinx of black quartz, judge my vow.'
        >>> deduper.add_document('myfile3', doc3)
        1
        >>> deduper.add_document('myfile2', doc)
        2
        >>> sorted(deduper.get_possible_duplicate_doc_names())
        [('myfile2', 'myfile1')]

        """
        col_dicts = [defaultdict(list) for idx in range(self.num_hash_fun)]  # type: ignore
        for row_idx, row in enumerate(self.hash_data):
            for col_idx, col_val in enumerate(row):
                col_dicts[col_idx][col_val].append((row_idx, col_idx))
        dupe_count = 0
        row_set = set()
        rows_to_compare = set()
        for coldict in col_dicts:
            for _, vals in coldict.items():
                if len(vals) > 1:
                    dupe_count += 1
                    rows = [row for row, col in vals]
                    to_eval = list(permutations(rows, 2))
                    for item in to_eval:
                        rows_to_compare.add(item)
                    for data in rows:
                        row_set.add(data)
        distinct_rows_to_compare = set()  # type: Set[Tuple[int, int]]
        for row1, row2 in rows_to_compare:
            if (row1, row2) not in distinct_rows_to_compare and (
                row2,
                row1,
            ) not in distinct_rows_to_compare:
                distinct_rows_to_compare.add((row1, row2))
        evaluations = [
            (self.idx_to_doc_name[row1], self.idx_to_doc_name[row2])
            for row1, row2 in distinct_rows_to_compare
            if sum(np.array(self.hash_data[row1]) == np.array(self.hash_data[row2]))
            / float(self.num_hash_fun)
            >= threshold
        ]  # type: List[Tuple[str, str]]
        return evaluations

    def calculate_similarity(self, text_one: str, text_two: str) -> float:
        """
        Calculate Jaccard similarity of the two texts
        https://en.wikipedia.org/wiki/Jaccard_index
        Note: accuracy suffers with very short text lengths.
        :param text_one:
        :param text_two:
        :return: float: 1 equals exact match

        >>> deduper = TextDeduplicater()
        >>> one = 'The quick brown fox jumped over the lazy dog.'
        >>> two = 'The silly dog ate the lazy fox.'
        >>> deduper.calculate_similarity(one, two)
        0.45454545454545453
        >>> deduper.calculate_similarity(one, one)
        1.0

        """
        if self.punctuation_substitutions:
            text_one = text_one.translate(self.punctuation_substitutions)
            text_two = text_two.translate(self.punctuation_substitutions)
        one_words = set()  # type: Set[str]
        two_words = set()  # type: Set[str]
        for tmp in text_one.split():
            one_words.add(tmp)
        for tmp in text_two.split():
            two_words.add(tmp)
        shared = len(one_words & two_words)
        union_vals = len(one_words | two_words)
        return shared / union_vals


def punctuation_for_spaces_dict() -> Dict[int, str]:
    """Provide a dictionary for removing punctuation, keeping spaces. Essential for scansion
    to keep stress patterns in alignment with original vowel positions in the verse.
    :return dict with punctuation from the unicode table

    >>> print("I'm ok! Oh #%&*()[]{}!? Fine!".translate(
    ... punctuation_for_spaces_dict()).strip())
    I m ok  Oh              Fine
    """
    return dict(
        (i, " ")
        for i in range(sys.maxunicode)
        if unicodedata.category(chr(i)).startswith("P")
    )


# pylint: disable=line-too-long
def grammify(word_list: List[str], num: int = 3):
    """
    Generate n-gram tuples from a word list.
    If not enough words, an empty list is returned.

    >>> word_list = ['Father','Christmas','Gave','Daddy','An','Electric','Blanket']
    >>> grammify(word_list, num=3)
    [['Father', 'Christmas', 'Gave'], ['Christmas', 'Gave', 'Daddy'], ['Gave', 'Daddy', 'An'], ['Daddy', 'An', 'Electric'], ['An', 'Electric', 'Blanket']]
    >>> grammify(word_list[5:], num=3)
    []
    """
    return [word_list[i : i + num] for i in range(len(word_list) - num + 1)]


def pick_random_coeffs(num: int, max_hash: int = MAX_SHINGLE_HASH) -> List[int]:
    """
    Generate a list of 'num' random coefficients for the random hash functions, while ensuring
    that the same value does not appear multiple times in the list.

    :param num: the number of random coefficients desired
    :param max_hash: the ceiling value of the random int range
    :return: a list of num random values

    >>> seeds = pick_random_coeffs(10)
    >>> len(set(seeds))
    10
    """
    rand_list = []  # type: List[int]
    while num > 0:
        rand_index = random.randint(0, max_hash)
        # Ensure all numbers are random
        while rand_index in rand_list:
            rand_index = random.randint(0, max_hash)
        rand_list.append(rand_index)
        num -= 1
    return rand_list
