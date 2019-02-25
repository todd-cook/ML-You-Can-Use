"""`corpus_fun.py` - functions for corpus processing."""
import logging

__author__ = 'Todd Cook <todd.g.cook@gmail.com>'
__license__ = 'MIT License'

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

# pylint: disable=line-too-long

def get_file_type_list(all_file_ids,
                       corpus_texts_by_type=None,
                       corpus_directories_by_type=None):
    """
    Match CorpusReader's fileids collection to document types.
    :param all_file_ids: a list of fileids, usually filenames
    :param corpus_texts_by_type: a dictionary mapping filenames keyed by type
    :param corpus_directories_by_type: a dictionary mapping directories, keyed by type
    :return: a list of tuples (filename, type of file)

    >>> files = ['./caesar/alex.txt', './caesar/bc1.txt', './caesar/bc2.txt', './caesar/bc3.txt', './caesar/bellafr.txt', './caesar/gall1.txt', './caesar/gall2.txt', './caesar/gall3.txt', './caesar/gall4.txt', './caesar/gall5.txt', './caesar/gall6.txt', './caesar/gall7.txt', './caesar/gall8.txt', './caesar/hisp.txt', './horace/arspoet.txt', './horace/carm1.txt', './horace/carm2.txt', './horace/carm3.txt', './horace/carm4.txt', './horace/carmsaec.txt', './horace/ep.txt', './horace/epist1.txt', './horace/epist2.txt', './horace/serm1.txt', './horace/serm2.txt']
    >>> sample_texts_by_type = {'republican': ['./caesar/bc1.txt', './caesar/bc2.txt', './caesar/bc3.txt'], 'augustan': ['./horace/carm1.txt', './horace/carm2.txt', './horace/carm3.txt', './horace/carm4.txt', './horace/carmsaec.txt']}
    >>> sample_texts_by_dir = {'republican': [ './caesar'], 'augustan': ['horace']}
    >>> get_file_type_list(files, sample_texts_by_type, sample_texts_by_dir)
    [('./caesar/bc1.txt', 'republican'), ('./caesar/bc2.txt', 'republican'), ('./caesar/bc3.txt', 'republican'), ('./horace/carm1.txt', 'augustan'), ('./horace/carm2.txt', 'augustan'), ('./horace/carm3.txt', 'augustan'), ('./horace/carm4.txt', 'augustan'), ('./horace/carmsaec.txt', 'augustan')]

    """
    clean_ids_types = []

    for key, valuelist in corpus_texts_by_type.items():
        for value in valuelist:
            if value in all_file_ids:
                clean_ids_types.append((value, key))

    for key, valuelist in corpus_directories_by_type.items():
        for value in valuelist:
            corrected_dir = value.replace('./', '')
            corrected_dir = '{}/'.format(corrected_dir)
            for name in all_file_ids:
                if name.startswith(corrected_dir):
                    clean_ids_types.append((name, key))
    clean_ids_types.sort(key=lambda x: x[0])
    return clean_ids_types
