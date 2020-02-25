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
"""`smart_lower_transformer.py` - Removes Sentence Style Capitalization."""

import logging
from typing import List, Any

from sklearn.base import BaseEstimator, TransformerMixin


LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

# pylint: disable=invalid-name,unused-argument


class SmartLowerTransformer(BaseEstimator, TransformerMixin):
    """
    Removes sentence style capitalization:
    Transform the first word of each sentence of a tokenized sentence matrix
    according to always observed lower case words.

    >>> smart_lower = SmartLowerTransformer(words_always_lower=['leuissima'])
    >>> corpus =  [['Leuissima', 'virum', 'Cano'],['perlucent', 'Arenas', 'Tum']]
    >>> smart_lower.transform(corpus)
    [['leuissima', 'virum', 'Cano'], ['perlucent', 'Arenas', 'Tum']]

    """

    def __init__(self, words_always_lower=None, lower_only_file=None):
        """

        :param lower_only_file:
        """
        self.words_always_lower = (
            set() if not words_always_lower else set(words_always_lower)
        )
        if lower_only_file:
            with open(lower_only_file, "rt", encoding="utf-8") as reader:
                for line in reader:
                    self.words_always_lower.add(line.strip())

    def fit(self, string_matrix: List[List[str]], y: List[Any] = None):
        """

        :param X:
        :param y:
        :return:
        """
        return self

    def _correct_word(self, idx: int, word: str) -> str:
        """
        Lower case first letter on a word with idx zero
        :param word:
        :return:

        >>> smart_lower = SmartLowerTransformer(words_always_lower=['sun'])
        >>> smart_lower._correct_word(0, 'Sun')
        'sun'

        >>> smart_lower._correct_word(1, 'Plato')
        'Plato'

        """
        if idx == 0 and word:
            candidate = word[0].lower() + word[1:]
            if candidate in self.words_always_lower:
                return candidate
        return word

    def transform(self, string_matrix: List[List[str]]) -> List[List[str]]:
        """
        Modify the matrix
        :param string_matrix:
        :return:
        """
        return [
            [self._correct_word(idx, word) for idx, word in enumerate(sentence)]
            for sentence in string_matrix
        ]
