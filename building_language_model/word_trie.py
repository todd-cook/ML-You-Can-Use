class WordTrie(object):
    """Keep track of whole words in a collection."""
    def __init__(self, word_ending_marker=None):
        self.root = dict()
        if not word_ending_marker:
            word_ending_marker = chr(3)
        self.EOT = word_ending_marker  # End of transmission, to mark word endings

    def add(self, word):
        curr_root = self.root
        for char in word:
            if not curr_root.get(char, None):
                curr_root[char] = dict()
            curr_root = curr_root[char]
        if not curr_root.get(self.EOT, None):
            curr_root[self.EOT] = dict()

    def has_word(self, word):
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
                    terminal = curr_root.get(char, None)
                    if terminal:
                        if self.EOT in terminal.keys():
                            return True
                    return False
                curr_root = curr_root[char]
            else:
                curr_root[char] = dict()
                curr_root = curr_root[char]
        if curr_root.get(self.EOT, None):
            return True
        return False
    
    def extract_word_pair(self, long_word, min_word_length=4):
        """4 characters = min word length to join; thus, we skip many short prepositions which often get added to verbs"""
        if len(long_word) < min_word_length * 2 or self.has_word(long_word):
            return [long_word]
        for idx in range(min_word_length, len(long_word) - min_word_length + 1): 
            word1 = long_word[:idx]
            word2 = long_word[idx:]
            if self.has_word(word1) and self.has_word(word2):
                return [word1, word2]
        return [long_word]  
