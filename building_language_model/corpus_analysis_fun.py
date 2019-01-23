"""`corpus_analysis_fun.py` - methods for analysing a corpus"""
import statistics
from collections import Counter, defaultdict

from aeoe_replacer import AEOEReplacer

from cltk.stem.latin.j_v import JVReplacer
from tqdm import tqdm


def get_word_lengths(corpus_reader, max_word_length=100):
    """Get the word length/frequency distribution"""
    word_lengths = Counter()
    jv_replacer = JVReplacer()
    files = corpus_reader.fileids()
    for file in tqdm(files, total=len(files), unit='files'):
        for word in corpus_reader.words([file]):
            word = jv_replacer.replace(word)
            word_length = len(word)
            if word.isalpha() and word_length <= max_word_length:
                word_lengths.update({word_length: 1})
    return word_lengths


def get_samples_for_lengths(corpus_reader, num_samples=5):
    """Get a number of sample words for each word length; good for sanity checking."""
    samples_lengths = defaultdict(list)
    jv_replacer = JVReplacer()
    for word in corpus_reader.words():
        if word.isalpha():
            word = jv_replacer.replace(word)
            word_length = len(word)
            samples_lengths[word_length].append(word)
            samples_lengths[word_length] = samples_lengths[word_length][
                                           :num_samples]  # trim to num_samples size
    return samples_lengths


def get_char_counts(corpus_reader):
    """Get a frequency distribution of characters in a corpus."""
    char_counter = Counter()
    files = corpus_reader.fileids()
    for file in tqdm(files, total=len(files), unit='files'):
        for word in corpus_reader.words([file]):
            if word.isalpha():
                for car in word:
                    char_counter.update({car: 1})
    return char_counter

def get_split_words (corpus_reader, word_trie):
    """
    Search a corpus for improperly joined words, defined by a discrete trie model.
    return a dictionary, keys are files, and values are lists of tuples of the split words.
    """
    split_words = defaultdict(list)
    jv_replacer = JVReplacer()
    aeoe_replacer = AEOEReplacer()
    files = corpus_reader.fileids()
    for file in tqdm(files, total=len(files), unit='files'):
        for word in corpus_reader.words([file]):
            word = aeoe_replacer.replace(jv_replacer.replace(word))
            if len(word) > 15 and not word_trie.has_word(word):
                word_list = word_trie.extract_word_pair(word)
                if len(word_list)==2:
                    split_words[file].append(word_list)
    return split_words

def get_mean_stdev(mycounter):
    """
    Return the mean, stdev for a counter of integers
    """
    all_lens = []
    for key in mycounter:
        all_lens += [key] * mycounter[key]
    return statistics.mean(all_lens), statistics.stdev(all_lens)