"""`trie_transformer.py` - Auto-splice improperly joined words in a tokenized sentence matrix."""

import datetime
import logging
import os
import pickle
from typing import List, Any

from sklearn.base import BaseEstimator, TransformerMixin
from mlyoucanuse.word_trie import WordTrie

__author__ = 'Todd Cook <todd.g.cook@gmail.com>'
__license__ = 'MIT License'

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


# pylint: disable=invalid-name,unused-argument

class TrieTransformer(BaseEstimator, TransformerMixin):
    """
    Auto splice improperly joined words in a tokenized sentence matrix.
    """

    def __init__(self, trie_file=None, word_list=None, save_unseen=True, save_dir=None):
        """

        :param trie_file:
        :param word_list:
        :param save_unseen:
        :param save_dir:

        >>> known_words = ['maturita','temperueniunt', 'radicibus', 'subministres']
        >>> trie_transformer = TrieTransformer(word_list=known_words)
        >>> corrupt_corpus = [['maturitatemperueniunt', 'est'], ['radicibussubministres', 'amo']]
        >>> trie_transformer.transform(corrupt_corpus)
        [['maturita', 'temperueniunt', 'est'], ['radicibus', 'subministres', 'amo']]

        """
        if trie_file:
            with open(trie_file, 'rb') as reader:
                self.trie = pickle.load(reader)
        if word_list:
            self.trie = WordTrie()
            for word in word_list:
                self.trie.add(word)
        if save_unseen:
            self.save_unseen = True
            self.unseen = []
        self.save_dir = save_dir

    def fit(self, string_matrix: List[List[str]], y: List[Any] = None):
        """
        Template method
        :param X:
        :param y:
        :return:
        """
        return self

    def extract_word_pair(self, long_word):
        """

        :param long_word:
        :return:
        """

        if self.trie.has_word(long_word):
            return [long_word]

        for idx in range(2, len(long_word) - 1):
            word1 = long_word[:idx]
            word2 = long_word[idx:]
            if self.trie.has_word(word1) and self.trie.has_word(word2):
                return [word1, word2]
        if self.save_unseen:
            self.unseen.append(long_word)
        return [long_word]  # don't swallow unknown words

    def transform(self, string_matrix: List[List[str]]) -> List[List[str]]:
        """
        Modify the matrix
        :param string_matrix:
        :return:
        """
        results = []
        if self.save_unseen:
            self.unseen = []
        for document in string_matrix:
            sentence = [] # type: List[str]
            for x in document:
                tmp_result = self.extract_word_pair(x)
                if tmp_result:
                    sentence += self.extract_word_pair(x)
            results.append(sentence)
        try:
            if self.save_unseen and self.save_dir:
                with open(os.path.join(self.save_dir,
                                       'unseen_words.{}.txt'.format(
                                           datetime.datetime.now().strftime('%Y.%m.%d'))),
                          'wt', encoding='utf8') as writer:
                    for word in self.unseen:
                        writer.write(word)
                        writer.write('\n')
        except OSError:
            LOG.exception('Failure in trying to save unseen words')
        return results
