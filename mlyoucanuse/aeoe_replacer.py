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
"""`aeoe_replacer.py` - Replace the single character ligatures with a multicharacter dipthong."""

import re
from typing import List
import logging


LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class AEOEReplacer:  # pylint: disable=too-few-public-methods
    """Replace 'œæ' with AE, 'Œ Æ' with OE.
    Classical Latin wrote the o and e separately (as has today again become the general practice),
     but the ligature was used by medieval and early modern writings, in part because the
     diphthongal sound had, by Late Latin, merged into the sound [e].
    See: https://en.wikipedia.org/wiki/%C5%92
    Æ (minuscule: æ) is a grapheme named æsc or ash, formed from the letters a and e, originally
     a ligature representing the Latin diphthong ae. It has been promoted to the full status of a
     letter in the alphabets of some languages, including Danish, Norwegian, Icelandic, and Faroese.
    See: https://en.wikipedia.org/wiki/%C3%86
    """

    def __init__(self):
        """Initialization for JVReplacer, reads replacement pattern tuple."""
        patterns = [(r"œ", "oe"), (r"æ", "ae"), (r"Œ", "OE"), (r"Æ", "AE")]
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        """Do character replacement."""
        for (pattern, repl) in self.patterns:
            text = re.subn(pattern, repl, text)[0]
        return text


def aeoe_transform(string_matrix: List[List[str]]) -> List[List[str]]:
    """
    A transformer function suitable for matrix sentence transformations.
    :param string_matrix: a data matrix: a list of a list of strings, each sublist a sentence.

    >>> aeoe_transform([[ 'pœma', 'cæsar', 'PŒMATA'], ['CÆSAR']])
    [['poema', 'caesar', 'POEMATA'], ['CAESAR']]
    """
    aeoe_replacer = AEOEReplacer()
    return [
        [aeoe_replacer.replace(word) for word in sentence] for sentence in string_matrix
    ]
