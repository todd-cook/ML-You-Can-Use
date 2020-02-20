"""`corpus_analysis_fun.py` - methods for analysing a corpus"""
import logging
import statistics
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

from nltk.corpus.reader.api import CorpusReader  # pylint: disable=no-name-in-module
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from mlyoucanuse.word_trie import WordTrie

__author__ = "Todd Cook <todd.g.cook@gmail.com>"
__license__ = "MIT License"

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def get_word_lengths(
    corpus_reader: CorpusReader, max_word_length: int = 100
) -> Dict[int, int]:
    """
    Get the word length/frequency distribution
    :param corpus_reader:
    :param max_word_length:
    :return:
    """
    word_lengths = Counter()  # type: Dict[int, int]
    files = corpus_reader.fileids()
    for file in tqdm(files, total=len(files), unit="files"):
        for word in corpus_reader.words(file):
            word_length = len(word)
            if word.isalpha() and word_length <= max_word_length:
                word_lengths.update({word_length: 1})
    return word_lengths


def get_samples_for_lengths(
    corpus_reader: CorpusReader, num_samples: int = 5
) -> Dict[int, List[str]]:
    """
    Get a number of sample words for each word length; good for sanity checking.
    :param corpus_reader:
    :param num_samples:
    :return:
    """
    samples_lengths = defaultdict(list)  # type: Dict[int, List[str]]
    files = corpus_reader.fileids()
    for file in tqdm(files, total=len(files), unit="files"):
        for word in corpus_reader.words(file):
            if word.isalpha():
                word_length = len(word)
                samples_lengths[word_length].append(word)
                samples_lengths[word_length] = samples_lengths[word_length][
                    :num_samples
                ]  # trim to num_samples size
    return samples_lengths


def get_char_counts(corpus_reader: CorpusReader) -> Dict[str, int]:
    """
    Get a frequency distribution of characters in a corpus.
    :param corpus_reader:
    :return:
    """
    char_counter = Counter()  # type: Dict[str, int]
    files = corpus_reader.fileids()
    for file in tqdm(files, total=len(files), unit="files"):
        for word in corpus_reader.words(file):
            if word.isalpha():
                for car in word:
                    char_counter.update({car: 1})
    return char_counter


def get_split_words(
    corpus_reader: CorpusReader, word_trie: WordTrie, max_word_length: int = 15
) -> Dict[str, List[str]]:
    """
    Search a corpus for improperly joined words, defined by a discrete trie model.
    return a dictionary, keys are files, and values are lists of tuples of the split words.

    :param corpus_reader:
    :param word_trie:
    :param max_word_length:
    :return:
    """
    split_words = defaultdict(list)  # type: Dict[str, List[str]]
    files = corpus_reader.fileids()
    for file in tqdm(files, total=len(files), unit="files"):
        for word in corpus_reader.words(file):
            if len(word) > max_word_length and not word_trie.has_word(word):
                word_list = word_trie.extract_word_pair(word)
                if len(word_list) == 2:
                    split_words[file] += word_list
    return split_words


def get_mean_stdev(mycounter: Dict[int, int]) -> Tuple[float, float]:
    """
    Return the mean, stdev for a counter of integers

    :param mycounter:
    :return:

    >>> counter = Counter(dict(zip(list(range(1, 10)), list(range(1, 10)))))
    >>> get_mean_stdev(counter)
    (6.333, 2.236)

    """
    all_lens = []  # type: List[int]
    for key in mycounter:
        all_lens += [key] * mycounter[key]
    return round(statistics.mean(all_lens), 3), round(statistics.stdev(all_lens), 3)


def create_probability_dist(
    mycounter: Dict[str, int], min_val: float = 0.000001, max_val: float = 0.999999
) -> Dict[str, int]:
    """
    Given a Counter object, create a dictionary of normalized probabilities.
    The default min value and max values are configurable. See:
    https://en.wikipedia.org/wiki/Cromwell%27s_rule

    :param mycounter:
    :param min_val:
    :param max_val:
    :return:

    >>> counter = Counter({'all': 1, 'work': 2, 'and': 4, 'no': 6, 'play': 9})
    >>> prob_map = create_probability_dist(counter)
    >>> prob_map['all']
    1.000000000001e-06
    >>> prob_map['play']
    0.9999989999999999

    """
    total_words = sum(mycounter.values())
    words = list(mycounter.keys())
    counts = np.array([tmp / float(total_words) for tmp in mycounter.values()])
    min_max_scaler = MinMaxScaler(feature_range=(min_val, max_val))
    scaled_data = min_max_scaler.fit_transform(counts.reshape(-1, 1))
    word_probabilities = {
        words[idx]: val[0] for idx, val in enumerate(scaled_data.tolist())
    }
    return word_probabilities
