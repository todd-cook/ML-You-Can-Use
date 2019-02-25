"""`pos_tag_fun.py` - functions for manipulating Part of Speech tags.

The POS tag definitions follow the usage found in the Perseus tagged texts.
for more info see: https://github.com/PerseusDL/treebank_data.git
"""

import logging
from typing import Dict, List  # pylint: disable=unused-import

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

POS = dict([  # part of speech, position 1
    ('N', 'noun'),
    ('V', 'verb'),
    ('A', 'adjective'),
    ('D', 'adverb'),
    ('C', 'conjunction'),
    ('R', 'preposition'),
    ('P', 'pronoun'),
    ('M', 'numeral'),
    ('I', 'interjection'),
    ('E', 'exclamation'),
    ('U', 'punctuation')])  # type: Dict[str, str]

PERSON = dict([  # person, position 2
    ('1', 'first person'),
    ('2', 'second person'),
    ('3', 'third person')])  # type: Dict[str, str]

NUMBER = dict([  # Grammatical number, position 3
    ('S', 'singular'),
    ('P', 'plural')])  # type: Dict[str, str]

TENSE = dict([  # Verb tense, position 4
    ('P', 'present'),
    ('I', 'imperfect'),
    ('R', 'perfect'),
    ('L', 'pluperfect'),
    ('T', 'future perfect'),
    ('F', 'future')])  # type: Dict[str, str]

MOOD = dict([  # Verb mood, position 5
    ('I', 'indicative'),
    ('S', 'subjunctive'),
    ('N', 'infinitive'),
    ('M', 'imperative'),
    ('P', 'participle'),
    ('D', 'gerund'),
    ('G', 'gerundive')])  # type: Dict[str, str]

VOICE = dict([  # Verb voice, position 6
    ('A', 'active'),
    ('P', 'passive'),
    ('D', 'deponent')])  # type: Dict[str, str]

GENDER = dict([  # Gender, position 7
    ('M', 'masculine'),
    ('F', 'feminine'),
    ('N', 'neuter')])  # type: Dict[str, str]

CASE = dict([  # Grammatical case, position 8
    ('N', 'nominative'),
    ('G', 'genitive'),
    ('D', 'dative'),
    ('A', 'accusative'),
    ('V', 'vocative'),
    ('B', 'ablative'),
    ('L', 'locative')])  # type: Dict[str, str]

DEGREE = dict([  # Degree, position 9
    ('P', 'apositive'),
    ('C', 'comparative'),
    ('S', 'superlative')])  # type: Dict[str, str]


def expand_postag(tag: str) -> List[str]:
    """
    >>> expand_postag('V-SPDANG-')
    ['verb', 'singular', 'present', 'gerund', 'active', 'neuter', 'genitive']

    >>> expand_postag('A-S---MA-')
    ['adjective', 'singular', 'masculine', 'accusative']

    """
    postaglist = (POS, PERSON, NUMBER, TENSE, MOOD, VOICE, GENDER, CASE, DEGREE)
    return [postaglist[idx][char]
            for idx, char in enumerate(tag)
            if char != '-']
