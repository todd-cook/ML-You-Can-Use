# Copyright 2020 Todd Cook
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`corpus_analysis_fun.py` - methods for analysing a corpus"""
import logging
from collections import Counter, defaultdict
from typing import List, Dict

from nltk.corpus.reader.api import CorpusReader  # pylint: disable=no-name-in-module
from tqdm import tqdm

from mlyoucanuse.word_trie import WordTrie


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
