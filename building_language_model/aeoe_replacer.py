"""`aeoe_replacer.py` - Replace the single charcter ligatures with multicharcter dipthong."""

import re
from typing import List

class AEOEReplacer(object):  # pylint: disable=R0903
    """Replace 'œæ' with AE, 'Œ Æ' with OE.
    Classical Latin wrote the o and e separately (as has today again become the general practice), but the ligature was used by medieval and early modern writings, in part because the diphthongal sound had, by Late Latin, merged into the sound [e].
    See: https://en.wikipedia.org/wiki/%C5%92
    Æ (minuscule: æ) is a grapheme named æsc or ash, formed from the letters a and e, originally a ligature representing the Latin diphthong ae. It has been promoted to the full status of a letter in the alphabets of some languages, including Danish, Norwegian, Icelandic, and Faroese.
    See:https://en.wikipedia.org/wiki/%C3%86
     """

    def __init__(self):
        """Initialization for JVReplacer, reads replacement pattern tuple."""
        patterns = [(r'œ', 'oe'), (r'æ', 'ae'), (r'Œ', 'OE'), (r'Æ', 'AE')]
        self.patterns = \
            [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        """Do character replacement"""
        for (pattern, repl) in self.patterns:
            text = re.subn(pattern, repl, text)[0]
        return text


def aeoe_transform(string_matrix: List[List[str]]) -> List[List[str]]:
    """

    :param string_matrix: a data matrix: a list wrapping a list of strings, with each sublist being a sentence.
    >>> aeoe_transform([[ 'pœma', 'cæsar', 'PŒMATA'], ['CÆSAR']])
    [['poema', 'caesar', 'POEMATA'], ['CAESAR']]
    """
    aeoe_replacer = AEOEReplacer()
    return [[aeoe_replacer.replace(word)
             for word in sentence]
            for sentence in string_matrix]