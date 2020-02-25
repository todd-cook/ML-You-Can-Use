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
"""`romanizer.py` - Romanizer - Transliteration class for Classical Greek."""

import logging
import re
from typing import List

from cltk.prosody.latin.scansion_constants import ScansionConstants


LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class Romanizer:
    """
    A class for Romanizing or Transliterating Ancient Greek into Latin.
    """

    def __init__(self):
        self.alternate_font_greek_to_roman = dict(
            [
                ("I", "i"),
                ("Α", "A"),
                ("Αι", "Ae"),
                ("Β", "B"),
                ("Γ", "G"),
                ("Δ", "D"),
                ("Ε", "E"),
                ("Ει", "E"),
                ("Ζ", "Z"),
                ("Η", "Ē"),
                ("Θ", "Th"),
                ("Ι", "I"),
                ("Κ", "K"),
                ("Λ", "L"),
                ("Μ", "M"),
                ("Ν", "N"),
                ("Ξ", "X"),
                ("Ο", "O"),
                ("Οι", "Oe"),
                ("Ου", "Ou"),
                ("Π", "P"),
                ("Ρ", "R"),
                ("Σ", "S"),
                ("Τ", "T"),
                ("Υ", "Y"),
                ("Υι", "Ui"),
                ("Φ", "Ph"),
                ("Χ", "Kh"),
                ("Ψ", "Ps"),
                ("Ω", "Ō"),
                ("α", "a"),
                ("αι", "ae"),
                ("β", "b"),
                ("γ", "g"),
                ("γγ", "ng"),
                ("γκ", "nc"),
                ("γξ", "nx"),
                ("γχ", "nch"),
                ("δ", "d"),
                ("ε", "e"),
                ("ει", "ei"),
                ("ζ", "z"),
                ("η", "ē"),
                ("θ", "th"),
                ("ι", "i"),
                ("κ", "k"),
                ("λ", "l"),
                ("μ", "m"),
                ("ν", "n"),
                ("ξ", "x"),
                ("ο", "o"),
                ("οι", "oe"),
                ("ου", "ou"),
                ("π", "p"),
                ("ρ", "r"),
                ("ς", "s"),
                ("σ", "s"),
                ("τ", "t"),
                ("υ", "u"),
                ("υι", "ui"),
                ("φ", "ph"),
                ("χ", "kh"),
                ("ψ", "ps"),
                ("ω", "ō"),
            ]
        )  # type: Dict[str, str]

        self.greek_to_roman = dict(
            [
                ("W", "V"),
                ("j", "i"),
                ("w", "v"),
                ("¯", " "),
                ("Ō", "O"),
                ("ʼ", " "),
                ("ʽ", " "),
                ("˘", " "),
                ("̀", " "),
                ("́", " "),
                ("̄", " "),
                ("̈", " "),
                ("̓", " "),
                ("̔", " "),
                ("͂", " "),
                ("ͅ", " "),
                ("ʹ", " "),
                ("Ά", "A"),
                ("ΐ", "i"),
                ("ά", "a"),
                ("έ", "e"),
                ("ή", "ē"),
                ("ί", "i"),
                ("ΰ", "u"),
                ("ϊ", "i"),
                ("ϋ", "u"),
                ("ό", "o"),
                ("ύ", "u"),
                ("ώ", "ō"),
                ("ϛ", "s"),
                ("ϝ", " "),
                ("ϟ", " "),
                ("ϡ", " "),
                ("ἀ", "a"),
                ("ἁ", "ha"),
                ("ἂ", "a"),
                ("ἃ", "ha"),
                ("ἄ", "a"),
                ("ἅ", "ha"),
                ("ἆ", "a"),
                ("ἇ", "ha"),
                ("Ἀ", "A"),
                ("Ἁ", "Ha"),
                ("Ἂ", "Ha"),
                ("Ἃ", "Ha "),
                ("Ἄ", "A"),
                ("Ἅ", "Ha"),
                ("Ἆ", "A"),
                ("ἐ", "e"),
                ("ἑ", "he"),
                ("ἒ", "e"),
                ("ἓ", "he"),
                ("ἔ", "e"),
                ("ἕ", "he"),
                ("Ἐ", "E"),
                ("Ἑ", "He"),
                ("Ἓ", "He"),
                ("Ἔ", "E"),
                ("Ἕ", "E"),
                ("ἠ", "ē"),
                ("ἡ", "hē"),
                ("ἢ", "ē"),
                ("ἣ", "hē"),
                ("ἤ", "ē"),
                ("ἥ", "hē"),
                ("ἦ", "ē"),
                ("ἧ", "hē"),
                ("Ἠ", "Ē"),
                ("Ἡ", "Hē"),
                ("Ἢ", "Hē"),
                ("Ἤ", "Ē"),
                ("Ἥ", "Ē"),
                ("Ἦ", "Ē"),
                ("Ἧ", "Hē"),
                ("ἰ", "i"),
                ("ἱ", "hi"),
                ("ἲ", "i"),
                ("ἳ", "hi"),
                ("ἴ", "i"),
                ("ἵ", "hi"),
                ("ἶ", "i"),
                ("ἷ", "hi"),
                ("Ἰ", "I"),
                ("Ἱ", "Hi"),
                ("Ἴ", "I"),
                ("Ἵ", "I"),
                ("Ἶ", "I"),
                ("ὀ", "o"),
                ("ὁ", "ho"),
                ("ὂ", "o"),
                ("ὃ", "ho"),
                ("ὄ", "o"),
                ("ὅ", "ho"),
                ("Ὀ", "O"),
                ("Ὁ", "Ho"),
                ("Ὃ", "Ho"),
                ("Ὄ", "O"),
                ("Ὅ", "Ho"),
                ("ὐ", "u"),
                ("ὑ", "hu"),
                ("ὒ", "u"),
                ("ὓ", "hu"),
                ("ὔ", "u"),
                ("ὕ", "hu"),
                ("ὖ", "u"),
                ("ὗ", "hu"),
                ("Ὑ", "Hy"),
                ("Ὕ", "Y"),
                ("Ὗ", "Hu"),
                ("ὠ", "ō"),
                ("ὡ", "hō"),
                ("ὢ", "ō"),
                ("ὣ", "ho"),
                ("ὤ", "ō"),
                ("ὥ", "hō"),
                ("ὦ", "ō"),
                ("ὧ", "hō"),
                ("Ὠ", "O"),
                ("Ὡ", "Hō"),
                ("Ὣ", "Hō"),
                ("Ὤ", "O"),
                ("Ὥ", "Ō"),
                ("Ὦ", "O"),
                ("Ὧ", "Hō"),
                ("ὰ", "a"),
                ("ά", "a"),
                ("ὲ", "e"),
                ("έ", "e"),
                ("ὴ", "ē"),
                ("ή", "e"),
                ("ὶ", "i"),
                ("ί", "i"),
                ("ὸ", "o"),
                ("ό", "o"),
                ("ὺ", "u"),
                ("ύ", "u"),
                ("ὼ", "ō"),
                ("ώ", "ō"),
                ("ᾀ", "a"),
                ("ᾁ", "ha"),
                ("ᾄ", "a"),
                ("ᾅ", "ha"),
                ("ᾆ", "a"),
                ("ᾇ", "ha"),
                ("ᾐ", "ē"),
                ("ᾑ", "hē"),
                ("ᾔ", "ē"),
                ("ᾕ", "hē"),
                ("ᾖ", "ē"),
                ("ᾗ", "hē"),
                ("ᾘ", "Ē"),
                ("ᾠ", "ō"),
                ("ᾡ", "Hō"),
                ("ᾤ", "ō"),
                ("ᾦ", "ō"),
                ("ᾧ", "hō"),
                ("ᾬ", "Ō"),
                ("ᾲ", "a"),
                ("ᾳ", "a"),
                ("ᾴ", "a"),
                ("ᾶ", "a"),
                ("ᾷ", "a"),
                ("᾿", " "),
                ("ῂ", "ē"),
                ("ῃ", "ē"),
                ("ῄ", "ē"),
                ("ῆ", "ē"),
                ("ῇ", "ē"),
                ("ῒ", "i"),
                ("ΐ", "i"),
                ("ῖ", "i"),
                ("ῗ", "i"),
                ("ῢ", "u"),
                ("ῤ", "r"),
                ("ῥ", "rh"),
                ("ῦ", "u"),
                ("Ῥ", "Rh"),
                ("ῲ", "ō"),
                ("ῳ", "ō"),
                ("ῴ", "ō"),
                ("ῶ", "ō"),
                ("ῷ", "ō"),
            ]
        )  # type: Dict[str, str]

        self.greek_to_roman_dipthongs = dict(
            [
                (" Ἥ", "Hē"),
                ("Αὖ", "Au"),
                ("Αἱ", "Hai"),
                ("Αὑ", "Hau"),
                ("Αὕ", "Hau"),
                ("Αὗ", "Hau"),
                ("Γγ", "Ng"),
                ("Ει", "Ei"),
                ("Εὖ", "Eu"),
                ("Εἵ", "Hei"),
                ("Εἶ", "Ei"),
                ("Εἷ", "Hei"),
                ("Εὑ", "Heu"),
                ("Εὔ", "Eu"),
                ("Οι", "Oi"),
                ("Ου", "Ou"),
                ("Οἱ", "Hoi"),
                ("Οἳ", "Hoi"),
                ("Οἵ", "Hoi"),
                ("Οἷ", "Hoi"),
                ("Οὑ", "Hou"),
                ("Οὓ", "Hou"),
                ("Οὕ", "Hou"),
                ("Οὗ", "Hou"),
                ("Υἱ", "Hui"),
                ("αἱ", "hai"),
                ("αὑ", "hau"),
                ("αὕ", "hau"),
                ("αὖ", "au"),
                ("αὗ", "hau"),
                ("γγ", "ng"),
                ("ει", "ei"),
                ("εἵ", "hei"),
                ("εἶ", "ei"),
                ("εἷ", "hei"),
                ("εὑ", "heu"),
                ("εὔ", "eu"),
                ("εὖ", "eu"),
                ("οι", "oi"),
                ("ου", "ou"),
                ("οἱ", "hoi"),
                ("οἳ", "hoi"),
                ("οἵ", "hoi"),
                ("οἷ", "hoi"),
                ("οὑ", "hou"),
                ("οὓ", "hou"),
                ("οὕ", "hou"),
                ("οὗ", "hou"),
                ("υἱ", "hui"),
            ]
        )  # type: Dict[str, str]

        scansion_constants = ScansionConstants()
        self.macrons_to_vowels = dict(
            zip(
                list(scansion_constants.ACCENTED_VOWELS),
                list(scansion_constants.VOWELS),
            )
        )  # type: Dict[str, str]

    def transliterate(self, word: str, demacronize: bool = False) -> str:
        """

        :param word:
        :return:

        >>> romanizer = Romanizer()
        >>> romanizer.transliterate('και το καλον και το αγαθον')
        'kai to kalon kai to agathon'
        >>> romanizer.transliterate(' Ἥραν')
        'Hēran'
        """
        for key in self.greek_to_roman_dipthongs:
            word = re.sub(key, self.greek_to_roman_dipthongs[key], word)
        word = "".join([self.greek_to_roman.get(letter, letter) for letter in word])
        word = "".join(
            [self.alternate_font_greek_to_roman.get(letter, letter) for letter in word]
        )
        if demacronize:
            return self.demacronize_text(word)
        return word

    def demacronize_text(self, text: str) -> str:
        """

        :param word: a string with macronized vowels
        :return: the string with macrons converted to regular cases

        >>> romanizer = Romanizer()
        >>> romanizer.demacronize_text('Ō Athēnaioi')
        'O Athenaioi'
        >>> romanizer.demacronize_text('Ō')
        'O'
        """
        for key, val in self.macrons_to_vowels.items():
            text = re.sub(key, val, text)
        return text


def romanizer_transform(string_matrix: List[List[str]]) -> List[List[str]]:
    """
    A transformer function suitable for matrix sentence transformations.
    :param string_matrix: a data matrix: a list of a list of strings, each sublist a sentence.

    >>> romanizer_transform([['και', 'το', 'καλον', 'και', 'το', 'αγαθον'], [' Ἥραν']])
    [['kai', 'to', 'kalon', 'kai', 'to', 'agathon'], ['Hēran']]
    """
    romanizer = Romanizer()
    return [
        [romanizer.transliterate(word) for word in sentence]
        for sentence in string_matrix
    ]
