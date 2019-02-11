"""`doc2tokens_transformer.py` - Transformer for Documents."""
import unicodedata

import re
from sklearn.base import BaseEstimator, TransformerMixin
from cltk.tokenize.word import WordTokenizer # pylint: disable=no-name-in-module
from cltk.stem.latin.j_v import JVReplacer
from cltk.tokenize.sentence import TokenizeSentence # pylint: disable=no-name-in-module
from cltk.prosody.latin.scansion_constants import ScansionConstants

from cltk.prosody.latin.string_utils import punctuation_for_spaces_dict

__author__ = 'Todd Cook <todd.g.cook@gmail.com>'
__license__ = 'MIT License'


class Doc2TokensTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer.
    """

    def __init__(self, language='latin', valid_chars=None, drop_regexes=None):
        """

        :param language:
        :param valid_chars:
        :param drop_regexes:
        """
        if valid_chars:
            self.valid_chars = valid_chars
        else:
            scansion_constants = ScansionConstants()
            self.valid_chars = set(scansion_constants.ACCENTED_VOWELS + scansion_constants.VOWELS +
                                   scansion_constants.CONSONANTS)
        self.replacer = JVReplacer()
        self.sentence_tokenizer = TokenizeSentence(language)
        self.word_tokenizer = WordTokenizer(language)
        self.punctuation_space_map = punctuation_for_spaces_dict()
        if not drop_regexes:
            drop_regexes = [r'[0-9]+[a-zA-Z]', r'\s+']
        self.drop_regexes = [re.compile(exp) for exp in drop_regexes]

    def has_all_valid_chars(self, tok):
        """

        :param tok:
        :return:
        """
        for car in tok:
            if car not in self.valid_chars:
                return False
        return True

    def clean_tokens(self, tok):
        """

        :param tok:
        :return:
        """
        tok = self.clean_text(tok)
        if not tok.isdigit():
            tok = self.replacer.replace(tok)
            if not is_punct(tok):
                if self.has_all_valid_chars(tok):
                    # tmp.append(tok)
                    return tok
        return ''

    def clean_text(self, text):
        """

        :param text:
        :return:
        """
        text = text.translate(self.punctuation_space_map)
        for exp in self.drop_regexes:
            text = exp.sub(' ', text)
        return text

    def normalize(self, document):
        """

        :param document:
        :return:
        """
        return [
            self.clean_tokens(token)
            for sentence in self.sentence_tokenizer.tokenize(document)
            for token in self.word_tokenizer.tokenize(sentence)
            if not is_punct(token) if len(self.clean_tokens(token)) > 0
        ]

    def fit(self, X, y=None):  # pylint: disable=unused-argument,invalid-name
        """
        Stub for common function interface.
        :param X:
        :param y:
        :return:
        """
        return self

    def transform(self, documents):
        """

        :param documents:
        :return:
        >>> dummyX = [['The quick fox. The lazy dog.'],
        ... ['Wait: godot. Look, sunshine!']]
        >>> newX = Doc2TokensTransformer().transform(dummyX)
        >>> print(list(newX))
        [['The', 'quick', 'fox', 'The', 'lazy', 'dog'], ['Wait', 'godot', 'Look', 'sunshine']]
        """

        for document in documents:
            if document and isinstance(document[0], str):
                yield self.normalize(document[0])


def is_punct(token):
    """

    :param token:
    :return:

    >>> is_punct(';')
    True
    >>> is_punct('a')
    False
    >>> is_punct('F')
    False
    >>> is_punct('?')
    True
    """
    return all(unicodedata.category(char).startswith('P') for char in token)
