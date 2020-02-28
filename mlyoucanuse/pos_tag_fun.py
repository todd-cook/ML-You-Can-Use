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
"""`pos_tag_fun.py` - functions for manipulating Part of Speech tags.

The POS tag definitions follow the usage found in the Perseus tagged texts.
for more info see: https://github.com/PerseusDL/treebank_data.git
"""

import logging
from typing import Dict, List  # pylint: disable=unused-import


LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

POS = dict(
    [  # part of speech, position 1
        ("N", "noun"),
        ("V", "verb"),
        ("A", "adjective"),
        ("D", "adverb"),
        ("C", "conjunction"),
        ("R", "preposition"),
        ("P", "pronoun"),
        ("M", "numeral"),
        ("I", "interjection"),
        ("E", "exclamation"),
        ("U", "punctuation"),
    ]
)  # type: Dict[str, str]

PERSON = dict(
    [  # person, position 2
        ("1", "first person"),
        ("2", "second person"),
        ("3", "third person"),
    ]
)  # type: Dict[str, str]

NUMBER = dict(
    [("S", "singular"), ("P", "plural")]  # Grammatical number, position 3
)  # type: Dict[str, str]

TENSE = dict(
    [  # Verb tense, position 4
        ("P", "present"),
        ("I", "imperfect"),
        ("R", "perfect"),
        ("L", "pluperfect"),
        ("T", "future perfect"),
        ("F", "future"),
    ]
)  # type: Dict[str, str]

MOOD = dict(
    [  # Verb mood, position 5
        ("I", "indicative"),
        ("S", "subjunctive"),
        ("N", "infinitive"),
        ("M", "imperative"),
        ("P", "participle"),
        ("D", "gerund"),
        ("G", "gerundive"),
    ]
)  # type: Dict[str, str]

VOICE = dict(
    [("A", "active"), ("P", "passive"), ("D", "deponent")]  # Verb voice, position 6
)  # type: Dict[str, str]

GENDER = dict(
    [("M", "masculine"), ("F", "feminine"), ("N", "neuter")]  # Gender, position 7
)  # type: Dict[str, str]

CASE = dict(
    [  # Grammatical case, position 8
        ("N", "nominative"),
        ("G", "genitive"),
        ("D", "dative"),
        ("A", "accusative"),
        ("V", "vocative"),
        ("B", "ablative"),
        ("L", "locative"),
    ]
)  # type: Dict[str, str]

DEGREE = dict(
    [  # Degree, position 9
        ("P", "apositive"),
        ("C", "comparative"),
        ("S", "superlative"),
    ]
)  # type: Dict[str, str]


def expand_postag(tag: str) -> List[str]:
    """
    >>> expand_postag('V-SPDANG-')
    ['verb', 'singular', 'present', 'gerund', 'active', 'neuter', 'genitive']

    >>> expand_postag('A-S---MA-')
    ['adjective', 'singular', 'masculine', 'accusative']

    """
    postaglist = (POS, PERSON, NUMBER, TENSE, MOOD, VOICE, GENDER, CASE, DEGREE)
    return [postaglist[idx][char] for idx, char in enumerate(tag) if char != "-"]


def _get_key_from_val(mydict, the_val):
    """
    Helper method.

    >>> _get_key_from_val( {1: 'one', 2: 'two'}, 'one')
    1
    """
    for key, val in mydict.items():
        if val == the_val:
            return key
    return None


def to_postag(description: str):
    """

    :param description: Comma separated string
    :return: The postag

    >>> to_postag('adjective, plural, feminine, ablative')
    'A-P---FB-'

    """
    chars = ["-"] * 9
    for item in description.split(","):
        item = item.strip().lower()
        if item in POS.values():
            chars[0] = _get_key_from_val(POS, item)
        if item in PERSON.values():
            chars[1] = _get_key_from_val(PERSON, item)
        if item in NUMBER.values():
            chars[2] = _get_key_from_val(NUMBER, item)
        if item in TENSE.values():
            chars[3] = _get_key_from_val(TENSE, item)
        if item in MOOD.values():
            chars[4] = _get_key_from_val(MOOD, item)
        if item in VOICE.values():
            chars[5] = _get_key_from_val(VOICE, item)
        if item in GENDER.values():
            chars[6] = _get_key_from_val(GENDER, item)
        if item in CASE.values():
            chars[7] = _get_key_from_val(CASE, item)
        if item in DEGREE.values():
            chars[8] = _get_key_from_val(DEGREE, item)
    return "".join(chars)
