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
"""`word_trie.py` - A trie that tracks word constructions."""
import logging
from typing import List, Dict, Tuple  # pylint: disable=unused-import

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class WordTrie:
    """Keep track of whole words in a collection."""

    def __init__(self, word_ending_marker: str = None):
        self.root = dict()  # type: Dict[str, str]
        if not word_ending_marker:
            word_ending_marker = chr(3)
        self.eot = word_ending_marker  # End of transmission, to mark word endings

    def add(self, word: str) -> None:
        """
        Add a word to the trie
        :param word:
        :return: None
        """
        curr_root = self.root
        for car in word:
            if not curr_root.get(car):
                curr_root[car] = {}  # type: ignore
            curr_root = curr_root[car]  # type: ignore
        if not curr_root.get(self.eot):
            curr_root[self.eot] = self.eot

    def add_all(self, words: List[str]) -> None:
        """
        Convenience method
        :param words:
        :return:
        """
        for word in words:
            self.add(word)

    def has_word(self, word: str) -> Tuple[bool, bool]:
        """
        Determine whether or not the exact word was pushed into this tree
        :param word: a string
        :return:  Tuple[Boolean, Boolean] - in_trie, is_terminal

        >>> mytrie = WordTrie()
        >>> mytrie.add('todd')
        >>> mytrie.has_word('todd')
        (True, True)
        >>> mytrie.has_word('to')
        (True, False)
        >>> mytrie.has_word('taco')
        (False, False)

        """
        curr_root = self.root
        depth = 0
        for car in word:
            if car in curr_root:
                depth += 1
                curr_root = curr_root[car]  # type: ignore
        in_tree = depth == len(word)
        is_terminal = self.eot in curr_root and in_tree
        return (in_tree, is_terminal)

    def extract_word_pair(self, long_word: str, min_word_length=4) -> List[str]:
        """
        4 characters = min word length to join; thus,
         we skip many short prepositions which often get added to verbs
        :param long_word:
        :param min_word_length:
        :return:

        >>> mytrie = WordTrie()
        >>> mytrie.add('todd')
        >>> mytrie.add('cook')
        >>> mytrie.extract_word_pair('toddcook')
        ['todd', 'cook']
        >>> mytrie.extract_word_pair('tacocat')
        ['tacocat']

        """
        _, is_terminal = self.has_word(long_word)
        if len(long_word) < min_word_length * 2 or is_terminal:
            return [long_word]
        for idx in range(min_word_length, len(long_word) - min_word_length + 1):
            word1 = long_word[:idx]
            word2 = long_word[idx:]
            _, is_terminal = self.has_word(word1)
            _, is_terminal2 = self.has_word(word2)
            if is_terminal and is_terminal2:
                return [word1, word2]
        return [long_word]


class TupleTrie:
    """A trie that tracks parts and wholes of data;
    typically letters and when letter combinations become complete words.
    """

    def __init__(self, word_ending_marker: str = None):
        self.root = dict()  # type: Dict[str, str]
        if not word_ending_marker:
            word_ending_marker = chr(3)
        self.eot = word_ending_marker  # End of transmission, to mark word endings

    def add(self, tup: Tuple[str, ...]) -> None:
        """
        Add a tuple to the trie
        :param tup:
        :return: None
        """
        curr_root = self.root
        for car in tup:
            if not curr_root.get(car):
                curr_root[car] = {}  # type: ignore
            curr_root = curr_root[car]  # type: ignore
        if not curr_root.get(self.eot):
            curr_root[self.eot] = self.eot  # ""

    def add_all(self, words: List[Tuple[str, ...]]) -> None:
        """
        Convenience method
        :param words:
        :return:
        """
        for tup in words:
            self.add(tup)

    def has_tuple(self, tup: Tuple[str, ...]) -> Tuple[bool, bool]:
        """
        Determine whether or not the exact tuple was pushed into this tree
        :param word: a string
        :return: Tuple[Boolean, Boolean] - in_trie, is_terminal

        >>> mytrie = TupleTrie()
        >>> mytrie.add( ('quick', 'brown', 'fox'))
        >>> mytrie.has_tuple( ('quick', 'brown', 'fox', 'farted'))
        (False, False)
        >>> mytrie.has_tuple( ('quick', 'brown', 'fox'))
        (True, True)
        >>> mytrie.add( ('todd', 'cook'))
        >>> mytrie.has_tuple( ('todd', 'cook'))
        (True, True)
        >>> mytrie.add( ('quick', 'brown', 'fox', 'jumped'))
        >>> mytrie.has_tuple( ('quick', 'brown'))
        (True, False)
        >>> mytrie.has_tuple( ('quick', 'brown', 'fart'))
        (False, False)
        >>> mytrie.has_tuple( ('quick', 'brown', 'fox', 'jumped'))
        (True, True)

        """
        curr_root = self.root
        depth = 0
        for word in tup:
            if word in curr_root:
                depth += 1
                curr_root = curr_root[word]  # type: ignore
        in_tree = depth == len(tup)
        is_terminal = self.eot in curr_root and in_tree
        return in_tree, is_terminal
