import unicodedata

import re
from sklearn.base import BaseEstimator, TransformerMixin
from cltk.tokenize.word import WordTokenizer
from cltk.stem.latin.j_v import JVReplacer
from cltk.tokenize.sentence import TokenizeSentence
from cltk.prosody.latin.scansion_constants import ScansionConstants

from cltk.prosody.latin.string_utils import punctuation_for_spaces_dict


class Doc2TokensTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, language='latin', valid_chars=None, drop_regexes=None):

        """

        :param language:
        :param valid_chars:
        :param drop_regexes:
        """
        if valid_chars:
            self.VALID_CHARS = valid_chars
        else:
            sc = ScansionConstants()
            self.VALID_CHARS = set(sc.ACCENTED_VOWELS + sc.VOWELS + sc.CONSONANTS)
        self.replacer = JVReplacer()
        self.sentence_tokenizer = TokenizeSentence(language)
        self.word_tokenizer = WordTokenizer(language)
        self.punctuation_space_map = punctuation_for_spaces_dict()
        if not drop_regexes:
            drop_regexes = [r'[0-9]+[a-zA-Z]', r'\s+']
        self.drop_regexes = [re.compile(exp) for exp in drop_regexes]

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def has_all_valid_chars(self, tok):
        for car in tok:
            if car not in self.VALID_CHARS:
                return False
        return True

    def clean_tokens(self, tok):
        tok = self.clean_text(tok)
        if not tok.isdigit():
            tok = self.replacer.replace(tok)
            if not self.is_punct(tok):
                if self.has_all_valid_chars(tok):
                    # tmp.append(tok)
                    return tok
        return ''

    def clean_text(self, text):
        text = text.translate(self.punctuation_space_map)
        for exp in self.drop_regexes:
            text = exp.sub(' ', text)
        return text

    def normalize(self, document):
        return [
            self.clean_tokens(token)
            for sentence in self.sentence_tokenizer.tokenize(document)
            for token in self.word_tokenizer.tokenize(sentence)
            if not self.is_punct(token)
            if len(self.clean_tokens(token)) > 0
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.normalize(document[0])
