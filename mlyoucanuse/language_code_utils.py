"""`language_code_utils.py` - utilty methods for dealing with fasttext language codes.
Some of the less common language code mappings were found in the iso639 library:
https://github.com/noumar/iso639
"""
import logging
from typing import Any, Tuple

from numpy import array

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def fast_text_prediction_to_three_letter_language_code(
    res: Tuple[Tuple[str, ...], Tuple[array]]
) -> Any:
    """Get a two letter language code from a fasttext language detection model.

    >>> import numpy as np
    >>> res = (('__label__eng', '__label__ron', '__label__vie'),
    ... np. array([9.99590933e-01, 2.40980095e-04, 1.49953354e-04]))
    >>> fast_text_prediction_to_three_letter_language_code(res)
    ['eng', 'ron', 'vie']
    >>> res = (('__label__oci',), np.array([0.88126439]))
    >>> fast_text_prediction_to_three_letter_language_code(res)
    'oci'
    >>> res = ([['__label__oci'], ['__label__tlh']], np.array([[0.88126439], [0.99645472]]))
    >>> fast_text_prediction_to_three_letter_language_code(res)
    ['oci', 'tlh']

    """
    if len(res[0]) == 1:
        label, _ = res
        return label[0][-3:]
    labels, _ = res
    if labels and isinstance(labels[0], list):
        labels = [tmp[0] for tmp in labels]
    return [tmp[-3:] for tmp in labels]


def fast_text_prediction_to_two_letter_language_code(
    res: Tuple[Tuple[str, ...], Tuple[array]]
) -> Any:
    """Get a two letter language code from a fasttext language detection model.

    >>> import numpy as np
    >>> res = (('__label__eng', '__label__ron', '__label__vie'),
    ... np.asarray([9.99590933e-01, 2.40980095e-04, 1.49953354e-04])
    ... ) # or res = mdl.predict('The quick brown fox', k=3)
    >>> fast_text_prediction_to_two_letter_language_code(res)
    ['en', 'ro', 'vi']
    >>> res = (('__label__eng',), np.array([0.99959093])
    ... ) # or res = mdl.predict('The quick brown fox')
    >>> fast_text_prediction_to_two_letter_language_code(res)
    'en'
    >>> res = ([['__label__oci'], ['__label__tlh']], np.array([[0.88126439], [0.99645472]]))
    >>> fast_text_prediction_to_two_letter_language_code(res)
    ['oc', 'Klingon']

    """
    if len(res[0]) == 1:
        label, _ = res
        return _get_name(label[0])
    labels, _ = res
    if labels and isinstance(labels[0], list):
        labels = [tmp[0] for tmp in labels]
    return [_get_name(tmp) for tmp in labels]


def _get_name(label):
    val = THREE_LETTER_TO_TWO_LETTER_LANGUAGE_CODES.get(label[-3:])
    if not val:
        LOG.warning("Language not found %s", label)
        return "Unknown"
    return val


THREE_LETTER_TO_TWO_LETTER_LANGUAGE_CODES = {
    "abk": "ab",  #: Abkhazian
    "aar": "aa",  #: Afar
    "afr": "af",  #: Afrikaans
    "alb": "sq",  #: Albanian
    "sqi": "sq",  #: Albanian
    "amh": "am",  #: Amharic
    "ara": "ar",  #: Arabic
    "arg": "an",  #: Aragonese
    "arm": "hy",  #: Armenian
    "hye": "hy",  #: Armenian
    "asm": "as",  #: Assamese
    "ave": "ae",  #: Avestan
    "aym": "ay",  #: Aymara
    "aze": "az",  #: Azerbaijani
    "bak": "ba",  #: Bashkir
    "baq": "eu",  #: Basque
    "eus": "eu",  #: Basque
    "bel": "be",  #: Belarusian
    "ben": "bn",  #: Bengali
    "bih": "bh",  #: Bihari
    "bis": "bi",  #: Bislama
    "bos": "bs",  #: Bosnian
    "bre": "br",  #: Breton
    "bul": "bg",  #: Bulgarian
    "bur": "my",  #: Burmese
    "mya": "my",  #: Burmese
    "cat": "ca",  #: Catalan
    "cha": "ch",  #: Chamorro
    "che": "ce",  #: Chechen
    "chi": "zh",  #: Chinese
    "zho": "zh",  #: Chinese
    "chu": "cu",  #: Church Slavic; Slavonic; Old Bulgarian
    "chv": "cv",  #: Chuvash
    "cor": "kw",  #: Cornish
    "cos": "co",  #: Corsican
    "scr": "hr",  #: Croatian
    "hrv": "hr",  #: Croatian
    "cze": "cs",  #: Czech
    "ces": "cs",  #: Czech
    "dan": "da",  #: Danish
    "div": "dv",  #: Dhivehi; Maldivian
    "dut": "nl",  #: Dutch
    "nld": "nl",  #: Dutch
    "dzo": "dz",  #: Dzongkha
    "eng": "en",  #: English
    "epo": "eo",  #: Esperanto
    "est": "et",  #: Estonian
    "fao": "fo",  #: Faroese
    "fij": "fj",  #: Fijian
    "fin": "fi",  #: Finnish
    "fre": "fr",  #: French
    "fra": "fr",  #: French
    "gla": "gd",  #: 'Gaelic': 'Scottish',  #:  Gaelic;
    "glg": "gl",  #: Galician
    "geo": "ka",  #: Georgian
    "kat": "ka",  #: Georgian
    "ger": "de",  #: German
    "deu": "de",  #: German
    "gre": "el",  #: Greek, Modern (1453-)
    "ell": "el",  #: Greek, Modern (1453-)
    "grn": "gn",  #: Guarani
    "guj": "gu",  #: Gujarati
    "hat": "ht",  #: Haitian; Haitian Creole
    "ina": "ia",
    #: Interlingua (International Auxiliary Language Association)
    "hau": "ha",  #: Hausa
    "heb": "he",  #: Hebrew
    "her": "hz",  #: Herero
    "hin": "hi",  #: Hindi
    "ho": "Motu",  #: Hiri
    "hun": "hu",  #: Hungarian
    "ice": "is",  #: Icelandic
    "isl": "is",  #: Icelandic
    "ido": "io",  #: Ido
    "ind": "id",  #: Indonesian
    #: 'Auxiliary': '(International',  #:  Interlingua
    "ile": "ie",  #: Interlingue
    "iku": "iu",  #: Inuktitut
    "ipk": "ik",  #: Inupiaq
    "gle": "ga",  #: Irish
    "ita": "it",  #: Italian
    "jpn": "ja",  #: Japanese
    "jav": "jv",  #: Javanese
    "kal": "kl",  #: Kalaallisut
    "kan": "kn",  #: Kannada
    "kas": "ks",  #: Kashmiri
    "kaz": "kk",  #: Kazakh
    "khm": "km",  #: Khmer
    "kik": "ki",  #: Kikuyu; Gikuyu
    "kin": "rw",  #: Kinyarwanda
    "kir": "ky",  #: Kirghiz
    "kom": "kv",  #: Komi
    "kor": "ko",  #: Korean
    "kua": "kj",  #: Kuanyama; Kwanyama
    "kur": "ku",  #: Kurdish
    "lao": "lo",  #: Lao
    "lat": "la",  #: Latin
    "lav": "lv",  #: Latvian
    "lim": "li",
    #: 'Limburgish': 'Limburger;',  #:  Limburgan;
    "lin": "ln",  #: Lingala
    "lit": "lt",  #: Lithuanian
    "ltz": "lb",  #: Luxembourgish; Letzeburgesch
    "mac": "mk",  #: Macedonian
    "mkd": "mk",  #: Macedonian
    "mlg": "mg",  #: Malagasy
    "may": "ms",  #: Malay
    "msa": "ms",  #: Malay
    "mal": "ml",  #: Malayalam
    "mlt": "mt",  #: Maltese
    "glv": "gv",  #: Manx
    "mao": "mi",  #: Maori
    "mri": "mi",  #: Maori
    "mar": "mr",  #: Marathi
    "mah": "mh",  #: Marshallese
    "mol": "mo",  #: Moldavian
    "mon": "mn",  #: Mongolian
    "nau": "na",  #: Nauru
    "nav": "nv",  #: Navajo ,  Navaho,
    "nde": "nd",  #: North Ndebele
    "nbl": "nr",  #: South Ndebele
    "ndo": "ng",  #: Ndonga
    "nep": "ne",  #: Nepali
    "sme": "se",  #: Northern Sami
    "nor": "no",  #: Norwegian
    "nob": "nb",  #: Norwegian Bokmal
    "nno": "nn",  #: Norwegian Nynorsk
    "nya": "ny",  #: 'Chewa': 'Chichewa;',  #:  Nyanja;
    "oci": "oc",  #: '1500);': '(post',  #:  Occitan
    "ori": "or",  #: Oriya
    "orm": "om",  #: Oromo
    "oss": "os",  #: Ossetian; Ossetic
    "sot": "st",  #: Sotho, Southern	st	sot
    "spa": "es",  #: Spanish; Castilian	es	spa
    "fry": "fy",  #: Western Frisian	fy	fry
    "zha": "za",  #: Zhuang; Chuang	za	zha
    "pli": "pi",  #: Pali
    "pan": "pa",  #: Panjabi
    "per": "fa",  #: Persian
    "fas": "fa",  #: Persian
    "pol": "pl",  #: Polish
    "por": "pt",  #: Portuguese
    "pus": "ps",  #: Pushto
    "que": "qu",  #: Quechua
    "roh": "rm",  #: Raeto-Romance
    "rum": "ro",  #: Romanian
    "ron": "ro",  #: Romanian
    "run": "rn",  #: Rundi
    "rus": "ru",  #: Russian
    "smo": "sm",  #: Samoan
    "sag": "sg",  #: Sango
    "san": "sa",  #: Sanskrit
    "srd": "sc",  #: Sardinian
    "scc": "sr",  #: Serbian
    "srp": "sr",  #: Serbian
    "sna": "sn",  #: Shona
    "iii": "ii",  #: Sichuan Yi
    "snd": "sd",  #: Sindhi
    "sin": "si",  #: Sinhala; Sinhalese
    "slo": "sk",  #: Slovak
    "slk": "sk",  #: Slovak
    "slv": "sl",  #: Slovenian
    "som": "so",  #: Somali
    "sun": "su",  #: Sundanese
    "swa": "sw",  #: Swahili
    "ssw": "ss",  #: Swati
    "swe": "sv",  #: Swedish
    "tgl": "tl",  #: Tagalog
    "tah": "ty",  #: Tahitian
    "tgk": "tg",  #: Tajik
    "tam": "ta",  #: Tamil
    "tat": "tt",  #: Tatar
    "tel": "te",  #: Telugu
    "tha": "th",  #: Thai
    "tib": "bo",  #: Tibetan
    "bod": "bo",  #: Tibetan
    "tir": "ti",  #: Tigrinya
    "ton": "to",  #: 'Islands)': '(Tonga',  #:  Tonga
    "tso": "ts",  #: Tsonga
    "tsn": "tn",  #: Tswana
    "tur": "tr",  #: Turkish
    "tuk": "tk",  #: Turkmen
    "twi": "tw",  #: Twi
    "uig": "ug",  #: Uighur
    "ukr": "uk",  #: Ukrainian
    "urd": "ur",  #: Urdu
    "uzb": "uz",  #: Uzbek
    "vie": "vi",  #: Vietnamese
    "vol": "vo",  #: Volapuk
    "wln": "wa",  #: Walloon
    "wel": "cy",  #: Welsh
    "cym": "cy",  #: Welsh
    "wol": "wo",  #: Wolof
    "xho": "xh",  #: Xhosa
    "yid": "yi",  #: Yiddish
    "yor": "yo",  #: Yoruba
    "zul": "zu",  #: Zulu
    # The following entries are less common:
    # pulled from the iso-639 library; they don't have two character language codes
    "ang": "Old English (ca. 450-1100)",
    "arq": "Algerian Arabic",
    "arz": "Egyptian Arabic",
    "avk": "Kotava",
    "bar": "Bavarian",
    "cbk": "Chavacano",
    "ceb": "Cebuano",
    "cho": "Choctaw",
    "cmn": "Mandarin Chinese",
    "crh": "Crimean Tatar",
    "csb": "Kashubian",
    "dsb": "Lower Sorbian",
    "dtp": "Central Dusun",
    "egl": "Emilian",
    "fkv": "Kven Finnish",
    "gcf": "Guadeloupean Creole French",
    "gos": "Gronings",
    "got": "Gothic",
    "grc": "Ancient Greek (to 1453)",
    "haw": "Hawaiian",
    "hoc": "Ho",
    "hrx": "Hunsrik",
    "hsb": "Upper Sorbian",
    "ilo": "Iloko",
    "jbo": "Lojban",
    "kab": "Kabyle",
    "kha": "Khasi",
    "krl": "Karelian",
    "kzj": "Coastal Kadazan",
    "lad": "Ladino",
    "ldn": "Láadan",
    "lfn": "Lingua Franca Nova",
    "lij": "Ligurian",
    "lvs": "Standard Latvian",
    "lzh": "Literary Chinese",
    "mhr": "Eastern Mari",
    "min": "Minangkabau",
    "mrj": "Western Mari",
    "nds": "Low German",
    "npi": "Nepali (individual language)",
    "nst": "Tase Naga",
    "ota": "Ottoman Turkish (1500-1928)",
    "pam": "Pampanga",
    "pcd": "Picard",
    "pes": "Iranian Persian",
    "pms": "Piemontese",
    "prg": "Prussian",
    "rif": "Tarifit",
    "rom": "Romany",
    "sah": "Yakut",
    "shs": "Shuswap",
    "shy": "Tachawit",
    "sux": "Sumerian",
    "swg": "Swabian",
    "swh": "Swahili (individual language)",
    "thv": "Tahaggart Tamahaq",
    "tlh": "Klingon",
    "tpi": "Tok Pisin",
    "tpw": "Tupí",
    "tzl": "Talossan",
    "war": "Waray (Philippines)",
    "wuu": "Wu Chinese",
    "zlm": "Malay (individual language)",
    "zsm": "Standard Malay",
    "zza": "Zaza",
}
