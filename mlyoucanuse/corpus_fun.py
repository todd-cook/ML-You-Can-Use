"""`corpus_fun.py` - functions for corpus processing."""

__author__ = 'Todd Cook <todd.g.cook@gmail.com>'
__license__ = 'MIT License'


def get_file_type_list(all_file_ids,
                       corpus_texts_by_type=None,
                       corpus_directories_by_type=None):
    """

    :param all_file_ids:
    :param corpus_texts_by_type:
    :param corpus_directories_by_type:
    :return:
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
