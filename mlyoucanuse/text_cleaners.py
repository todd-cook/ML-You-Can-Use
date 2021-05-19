# Copyright 2021 Todd Cook
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
"""`text_cleaners.py` - functions useful for cleaning text.
## Text Cleaning
from http://udallasclassics.org/wp-content/uploads/maurer_files/APPARATUSABBREVIATIONS.pdf
[...]  Square brackets, or in recent editions wavy brackets ʺ{...}ʺ,
 enclose words etc. that an editor thinks should be deleted (see ʺdel.ʺ)
  or marked as out of place (see ʺsecl.ʺ).
[...]     Square  brackets  in  a  papyrus  text, or in an  inscription,
  enclose  places  where words have been lost through physical damage.
   If this happens in mid-line, editors  use  ʺ[...]ʺ.
   If  only  the  end  of  the  line  is  missing,
     they  use  a  single  bracket ʺ[...ʺ
     If  the  lineʹs  beginning  is  missing,  they  use  ʺ...]ʺ
     Within  the  brackets, often each dot represents one missing letter.
[[...]] Double brackets enclose letters or words deleted by the medieval copyist himself.
(...)  Round  brackets  are  used  to  supplement  words  abbreviated  by  the  original  copyist; e.g. in an inscription: ʺtrib(unus) mil(itum) leg(ionis) IIIʺ
<...> diamond  (  =  elbow  =  angular)  brackets  enclose  words  etc.
 that  an  editor  has  added (see ʺsuppl.ʺ)
†   An obelus (pl. obeli) means that the word(s etc.) is very plainly corrrupt,
 but the editor  cannot  see  how  to  emend.
 If  only  one  word  is  corrupt,  there  is  only  one obelus,
 which precedes the word; if two or more words are corrupt, two obeli  enclose  them.
    (Such  at  least  is  the  rule--but  that  rule  is  often  broken,  especially  in  older  editions,  which  sometimes  dagger  several  words  using  only one obelus.)  To dagger words in this way is to ʺobelizeʺ them.
"""

import re


BRACE_STRIP = re.compile(r"{[^}]+}")
NUMERALS = re.compile(r"[0-9]+")
PUNCT = re.compile(r"[\\/':;,!\?\._『-]+")
QUOTES = re.compile(r'["”“]+')
# we will extract content between brackets, it is editorial
ANGULAR_BRACKETS = re.compile(r"([a-zA-Z]+)?<[a-zA-Z\s]+>([,\?\.a-zA-Z]+)?")
SQUARE_BRACKETS = re.compile(r"\[[^\]]+\]")
OBELIZED_WORDS = re.compile(r"†[^†]+†")
OBELIZED_WORD = re.compile(r"†[^\s]+\s")
OBELIZED_PLUS_WORDS = re.compile(r"[\+][^\+]+[\+]")
OBELIZED_PLUS_WORD = re.compile(r"[\+][^\s]+\s")
HYPHENATED = re.compile(r"\s[^-]+-[^-]+\s")


def dehyphenate(text):
    """
    Remove hyphens from text; used on texts that have an line breaks with hyphens
    that may creep into the text. Caution using this elsewhere.
    :param text:
    :return:

    >>> dehyphenate('quid re-tundo hier')
    'quid retundo hier'
    """

    idx_to_omit = []
    for item in HYPHENATED.finditer(text):
        idx_to_omit.insert(0, item.span())
    for start, end in idx_to_omit:
        text = text[:start] + text[start:end].replace("-", "") + text[end:]
    return text


def swallow(text, pattern_matcher):
    """

    :param text:
    :param pattern_matcher:
    :return:
    """
    idx_to_omit = []
    for item in pattern_matcher.finditer(text):
        idx_to_omit.insert(0, item.span())
    for start, end in idx_to_omit:
        text = text[:start] + text[end:]
    return text.strip()


def swallow_braces(text):
    """

    :param text:
    :return:

    >>> swallow_braces("{PRO P. QVINCTIO ORATIO} Quae res in civitate {etc}... ")
    'Quae res in civitate ...'

    """
    return swallow(text, BRACE_STRIP)


def drop_punct(text):
    """

    :param text:
    :return:

    >>> drop_punct('re-tundo')
    're tundo'
    """
    text = NUMERALS.sub(" ", text)
    text = HYPHENATED.sub("", text)
    text = PUNCT.sub(" ", text)
    text = QUOTES.sub(" ", text)
    # todo drop roman NUMERALS
    return text


# pylint: disable=too-many-statements
def normalize_accents(text:str)->str:
    """
    Remove accents
    :param text: text with undesired accents
    :return: clean text

    >>> normalize_accents('suspensám')
    'suspensam'
    >>> normalize_accents('quăm')
    'quam'
    >>> normalize_accents('aegérrume')
    'aegerrume'
    >>> normalize_accents('ĭndignu')
    'indignu'
    >>> normalize_accents('îs')
    'is'
    >>> normalize_accents('óccidentem')
    'occidentem'
    >>> normalize_accents('frúges')
    'fruges'

    """
    text = text.replace(r"á", "a")  # suspensám
    text = text.replace(r"Á", "A")
    text = text.replace(r"á", "a") # Note: this accent is different than the one above!
    text = text.replace(r"Á", "A")
    text = text.replace(r"ă", "a")  # 'quăm'
    text = text.replace(r"Ă", "A")
    text = text.replace(r"à", "a")
    text = text.replace(r"À", "A")
    text = text.replace(r"â", "a")
    text = text.replace(r"Â", "A")
    text = text.replace(r"ä", "a")
    text = text.replace(r"Ä", "A")
    text = text.replace(r"é", "e")  # aegérrume
    text = text.replace(r"è", "e")
    text = text.replace(r"È", "E")
    text = text.replace(r"é", "e")
    text = text.replace(r"É", "E")
    text = text.replace(r"ê", "e")
    text = text.replace(r"Ê", "E")
    text = text.replace(r"ë", "e")
    text = text.replace(r"Ë", "E")
    text = text.replace(r"ĭ", "i")  # ĭndignu
    text = text.replace(r"î", "i")  # 'îs'
    text = text.replace(r"í", "i")
    text = text.replace(r"í", "i")
    text = text.replace(r"î", "i")
    text = text.replace(r"Î", "I")
    text = text.replace(r"ï", "i")
    text = text.replace(r"Ï", "I")
    text = text.replace(r"ó", "o")  # óccidentem
    text = text.replace(r"ô", "o")
    text = text.replace(r"Ô", "O")
    text = text.replace(r"ö", "o")
    text = text.replace(r"Ö", "O")
    text = text.replace(r"û", "u")
    text = text.replace(r"Û", "U")
    text = text.replace(r"ù", "u")
    text = text.replace(r"Ù", "U")
    text = text.replace(r"ü", "u")
    text = text.replace(r"Ü", "U")
    text = text.replace(r"ú", "u")  # frúges
    text = text.replace(r"ÿ", "y")
    text = text.replace(r"Ÿ", "Y")
    text = text.replace(r"ç", "c")
    text = text.replace(r"Ç", "C")
    text = text.replace(r"ë", "e")
    text = text.replace(r"Ë", "E")
    text = text.replace(r"Ȳ", "Y")
    text = text.replace(r"ȳ", "y")
    return text


def remove_macrons(text:str)->str:
    """
    Remove macrons above vowels
    :param text: text with macronized vowels
    :return: clean text

    >>> remove_macrons("canō")
    'cano'
    >>> remove_macrons("Īuliī")
    'Iulii'

    """
    text = text.replace(r"ā", "a")
    text = text.replace(r"Ā", "A")
    text = text.replace(r"ē", "e")
    text = text.replace(r"Ē", "E")
    text = text.replace(r"ī", "i")
    text = text.replace(r"Ī", "I")
    text = text.replace(r"ō", "o")
    text = text.replace(r"Ō", "O")
    text = text.replace(r"ū", "u")
    text = text.replace(r"Ū", "U")
    return text


def swallow_angular_brackets(text):
    """
    >>> text= " <O> mea dext<e>ra illa CICERO RUFO Quo<quo>. modo proficiscendum <in> tuis.  deesse HS <c> quae    metu <exagitatus>, furore   <es>set consilium  "
    >>> swallow_angular_brackets(text)
    'mea  illa CICERO RUFO  modo proficiscendum  tuis.  deesse HS  quae    metu  furore    consilium'

    """
    text = swallow(text, ANGULAR_BRACKETS)
    # There are occasionally some unmatched ANGULAR_BRACKETS: TODO fix better
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    return text


def disappear_angle_brackets(text):
    """

    :param text:
    :return:
    """
    text = text.replace("<", "")
    text = text.replace(">", "")
    return text


def swallow_square_brackets(text):
    """

    :param text:
    :return:

    >>> swallow_square_brackets("qui aliquod institui[t] exemplum")
    'qui aliquod institui exemplum'
    >>> swallow_square_brackets("posthac tamen cum haec [tamen] quaeremus,")
    'posthac tamen cum haec  quaeremus,'

    """
    return swallow(text, SQUARE_BRACKETS)


def swallow_obelized_words(text):
    """

    :param text:
    :return:

    >>> swallow_obelized_words("tu Fauonium †asinium† dicas")
    'tu Fauonium  dicas'
    >>> swallow_obelized_words("tu Fauonium †asinium dicas")
    'tu Fauonium dicas'
    >>> swallow_obelized_words("meam +similitudinem+")
    'meam'
    >>> swallow_obelized_words("mea +ratio non habet" )
    'mea non habet'

    """
    text = swallow(text, OBELIZED_WORDS)
    text = swallow(text, OBELIZED_WORD)
    text = swallow(text, OBELIZED_PLUS_WORDS)
    return swallow(text, OBELIZED_PLUS_WORD)


def disappear_round_brackets(text):
    """

    :param text:
    :return:

    >>> disappear_round_brackets("trib(unus) mil(itum) leg(ionis) III")
    'tribunus militum legionis III'
    """
    text = text.replace("(", "")
    return text.replace(")", "")


def swallow_editorial(text):
    """

    :param text:
    :return:
    """
    text = disappear_round_brackets(text)
    text = swallow_angular_brackets(text)
    text = swallow_square_brackets(text)
    text = swallow_obelized_words(text)
    return text


def accept_editorial(text):
    """

    :param text:
    :return:
    """
    text = swallow_braces(text)
    text = disappear_round_brackets(text)
    text = swallow_obelized_words(text)
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("<", "")
    text = text.replace(">", "")
    text = text.replace("...", " ")
    return text


def truecase(word, case_counter):
    """
    Truecase

    :param word:
    :param case_counter:
    :return:
    """
    lcount = case_counter.get(word.lower(), 0)
    ucount = case_counter.get(word.upper(), 0)
    tcount = case_counter.get(word.title(), 0)
    if lcount == 0 and ucount == 0 and tcount == 0:
        return word  #: we don't have enough information to change the case
    if tcount > ucount and tcount > lcount:
        return word.title()
    if lcount > tcount and lcount > ucount:
        return word.lower()
    if ucount > tcount and ucount > lcount:
        return word.upper()
    return word
