"""`word_trie.py` - A trie that tracks word constructions."""
import logging
from typing import List, Dict  # pylint: disable=unused-import

__author__ = 'Todd Cook <todd.g.cook@gmail.com>'
__license__ = 'MIT License'

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
            curr_root[self.eot] = ''

    def add_all(self, words: List[str]) -> None:
        """
        Convenience method
        :param words:
        :return:
        """
        for word in words:
            self.add(word)

    def has_word(self, word: str) -> bool:
        """
        Determine whether or not the exact word was pushed into this tree
        :param word: a string
        :return: Boolean

        >>> mytrie = WordTrie()
        >>> mytrie.add('todd')
        >>> mytrie.has_word('todd')
        True
        >>> mytrie.has_word('to')
        False
        """
        curr_root = self.root
        for idx, char in enumerate(word):
            if char in curr_root:
                if idx + 1 == len(word):
                    terminal = curr_root.get(char)
                    if terminal:
                        if self.eot in terminal:
                            return True
                    return False
                curr_root = curr_root[char]  # type: ignore
            else:
                curr_root[char] = dict()  # type: ignore
                curr_root = curr_root[char]  # type: ignore
        if curr_root.get(self.eot):
            return True
        return False

    def extract_word_pair(self, long_word: str,
                          min_word_length=4) -> List[str]:
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
        if len(long_word) < min_word_length * 2 or self.has_word(long_word):
            return [long_word]
        for idx in range(min_word_length,
                         len(long_word) - min_word_length + 1):
            word1 = long_word[:idx]
            word2 = long_word[idx:]
            if self.has_word(word1) and self.has_word(word2):
                return [word1, word2]
        return [long_word]
