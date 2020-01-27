"""`embeddings.py` - utility methods for working with embeddings."""

import os
import gzip
import logging
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import numpy as np
from numpy import ndarray
from gensim import models
from keras.layers import Embedding
from keras.utils import get_file
from binaryornot.check import is_binary

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

EMBEDDINGS_METADATA = {
    # Wikipedia 2014 + Gigaword 5
    # (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download):
    # glove.6B.zip
    'glove': ['http://nlp.stanford.edu/data/glove.6B.zip', 'glove.6B.{}d.txt'],
    'GoogleNews': [
        'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',
        'GoogleNews-vectors-negative300.bin.gz'],
    # Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download):
    # glove.840B.300d.zip
    'CommonCrawl.840B': [
        'http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip', 'glove.840B.300d.txt'],
    # Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 200d vectors, 1.42 GB download):
    # glove.twitter.27B.zip
    'Twitter2B': ['http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip',
                  'glove.twitter.27B.{}d.txt'],
    #     Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download):
    #     glove.42B.300d.zip
    'CommonCrawl.42B': ['http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip',
                        'glove.42B.300d.txt']
}  # type: Dict[str, List[str]]


def decompress(the_filepath: Path) -> None:
    """
    Decompress a gzip file, unless it's already decompressed.
    :param the_filepath:
    :return: None
    """
    filepath = str(the_filepath)
    filepath_wo_ext = filepath[:filepath.rfind('.')]
    LOG.info('decompressing %s to %s', filepath, filepath_wo_ext)
    if os.path.exists(filepath_wo_ext):
        LOG.warning('File %s already exists', filepath_wo_ext)
        return
    with gzip.open(filepath, 'rb') as f_in:
        with open(filepath_wo_ext, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


# pylint: disable=too-many-arguments,too-many-locals

def get_embeddings_index(embedding_name: str,
                         url: str = None,
                         embeddings_filename: str = None,
                         parent_dir: str = None,
                         cache_dir: str = None,
                         embedding_dimensions: int = 300) -> Dict[str, ndarray]:
    """
    High level function for get an embedding index, usually from a public url, downloading and
    caching locally.

    :param embedding_name: the name of the embeddings, used to look up metadata values
    :param url: the URL where the embeddings may be found; this parameter overrides the baked in
    metadata paths
    :param embeddings_filename: the filename; usually appended onto the url
    :param parent_dir: where to store the files locally, if not specified then the keras cache
    directory will be used.
    :param cache_dir: where to store the files locally, if parent_dir is not specified then the
    keras cache directory will be used.
    :param embedding_dimensions: integer: 300 or 100, 50 etc
    :return: a dictionary of strings to ndarray of embedding values

    """
    file_template = ''
    if embedding_name in EMBEDDINGS_METADATA:
        url, file_template = EMBEDDINGS_METADATA[embedding_name]

    if not cache_dir and parent_dir:
        cache_dir = os.path.join(parent_dir, 'data', embedding_name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    parts = urlparse(str(url))
    filename = parts.path.split('/')[-1]

    parent_path = Path(str(cache_dir))
    embeddings_dir = parent_path / 'datasets'

    if embeddings_filename:
        embeddings_file = embeddings_filename
    else:
        if '{' in file_template and '}' in file_template:
            embeddings_file = file_template.format(embedding_dimensions)
        else:
            embeddings_file = file_template
    embed_file = embeddings_dir / embeddings_file
    if not embed_file.exists():  # if not exists, fetch
        LOG.info('initializing, please wait.')
        data_archive = get_file(
            fname=filename,
            origin=url,
            cache_dir=cache_dir,
            untar=False,
            extract=True)  # pylint disable:unused-variable
        LOG.info('Done initializing')
        if not data_archive:
            LOG.warning('Fail in fetch')
    embeddings_index = load_embeddings(str(embed_file), embedding_dimensions=embedding_dimensions)
    return embeddings_index

def read_text_embeddings(embedding_file: str,
                         embedding_dimensions: int = 300) -> Tuple[Dict[str, int], List[ndarray]]:
    """
    :param embedding_file:
    :param embedding_dimensions:
    :return:
    """
    embeddings_index = [] # type: List[ ndarray]
    word_positions = {} # Type: Dict[str, int]
    with open(embedding_file) as the_file:
        for idx, line in enumerate(the_file):
            # if header, skip first line
            values = line.rsplit(maxsplit=embedding_dimensions)
            word = values[0]
            word_positions[word] = idx
            matrix_row = np.asarray(values[1:], dtype='float32')
            assert embedding_dimensions == len(matrix_row)
            embeddings_index.append(matrix_row)
    return word_positions, embeddings_index


def load_embeddings(embedding_file: str, embedding_dimensions: int = 300) -> Dict[str, ndarray]:
    """
    Low level function for loading embeddings from an accessible path.

    :param embedding_file: valid full path to embeddings file
    :return: a dictionary mapping strings to an ndarray of embedding values

    >>> import tempfile, os
    >>> _, tmp_file = tempfile.mkstemp()
    >>> test_file = open(tmp_file, mode='wt')
    >>> _ = test_file.write('{} {}'.format('quick',
    ... np.array2string(np.random.rand(10)).replace('\\n', '')[1:-1]))
    >>> test_file.close()
    >>> embed_map = load_embeddings(test_file.name, embedding_dimensions=10)
    >>> type(list(embed_map.keys())[0]) == str
    True
    >>> type(list(embed_map.values())[0]) == ndarray
    True
    >>> os.unlink(test_file.name)

    """
    embeddings_index = {}  # type: Dict[str, ndarray]
    if not is_binary(embedding_file):
        positions, embeddings_matrix = read_text_embeddings(embedding_file, embedding_dimensions)
        assert len(embeddings_matrix[0]) == embedding_dimensions
        for word in positions:
            embeddings_index[word] = embeddings_matrix[positions[word]]
    else:
        try:
            keyed_vectors = models.KeyedVectors.load(embedding_file)
        except pickle.UnpicklingError:
            LOG.exception('failure using KeyedVectores.load, trying word2vec format')
            keyed_vectors = models.KeyedVectors.load_word2vec_format(embedding_file, binary=True,
                                                                     unicode_errors='replace')
        for word in keyed_vectors.vocab:
            embeddings_index[word] = keyed_vectors[word]
    LOG.info('Found %s word vectors.', len(embeddings_index))
    return embeddings_index


def create_embeddings_matrix(embeddings_index: Dict[str, ndarray],
                             vocabulary: Dict[str, int],
                             embeddings_dimensions: int = 100,
                             init_by_variance: bool = True) -> ndarray:
    """
    Create an embedding matrix

    :param embeddings_index: a dictionary mapping words to ndarrays
    :param vocabulary: a dictionary of unique words and their unique integer values
    :param embeddings_dimension: the dimensions of the embeddings
    :param init_by_variance: boolean; initialize matrix to random values with each column
    initialized to the corresponding the variance of the seed embeddings columns.
    :return: an integer index ordered matrix of the filtered embedding values.

    >>> myembeddings_idxs = {'quick': np.ones(10), 'brown' : np.zeros(10)}
    >>> vocab = {'the': 1, 'quick': 2, 'brown': 3, 'fox': 4}
    >>> embeds = create_embeddings_matrix(embeddings_index=myembeddings_idxs, vocabulary=vocab,
    ... embeddings_dimensions=10, init_by_variance=False)
    >>> len(embeds)
    5
    >>> np.array_equal(embeds[0], np.ones(10))
    False
    >>> np.array_equal(embeds[2], np.ones(10))
    True
    >>> np.array_equal(embeds[3], np.zeros(10))
    True

    """
    # pylint: disable=no-member
    if init_by_variance:
        cols = list(embeddings_index.values())[0].shape[0]
        embed_ar = np.asarray(list(embeddings_index.values()))
        matrix_variance = np.asarray(
            [np.var(embed_ar[:, idx]) for idx in range(cols)])
        del embed_ar
        embeddings_matrix = matrix_variance * np.random.rand(
            len(vocabulary) + 1, embeddings_dimensions)
    else:
        embeddings_matrix = np.random.rand(
            len(vocabulary) + 1, embeddings_dimensions)
    for word, idx in vocabulary.items():
        embedding_vector = embeddings_index.get(word)  # type: ignore
        if embedding_vector is not None:
            embeddings_matrix[idx] = embedding_vector
    LOG.info('Embeddings matrix shape: %s', embeddings_matrix.shape)
    return embeddings_matrix


def get_embeddings_layer(embeddings_matrix: ndarray,
                         name: str,
                         max_len: int,
                         trainable=False) -> Embedding:
    """
    Create an embedding layer

    :param embeddings_matrix: the integer indexed ndarray of embedding values
    :param name: the name to give the Embeddings layer; used for reference and display
    :param max_len: the max sequence length input
    :param trainable: whether or not to freeze the embeddings, or to allow them to be updated
    :return: Embedding layer

    >>> myembeddings_idxs = {'quick': np.ones(10), 'brown' : np.zeros(10)}
    >>> vocab = {'the': 1, 'quick': 2, 'brown': 3, 'fox': 4}
    >>> embeds = create_embeddings_matrix(embeddings_index=myembeddings_idxs, vocabulary=vocab,
    ... embeddings_dimensions=10, init_by_variance=False)
    >>> myembeds_layer = get_embeddings_layer(embeds, 'myembeds', max_len=20, trainable=False)
    >>> type(myembeds_layer) == Embedding
    True

    """
    embedding_layer = Embedding(
        input_dim=embeddings_matrix.shape[0],
        output_dim=embeddings_matrix.shape[1],
        input_length=max_len,
        weights=[embeddings_matrix],
        trainable=trainable,
        name=name)
    return embedding_layer


def create_embeddings_layer(embeddings_file: str,
                            embeddings_dimensions: int,
                            max_len: int,
                            layer_name: str,
                            word_index_map: Dict[str, int],
                            trainable: bool = True):
    """
    Simplified helper method
    :param embeddings_file:
    :param embeddings_dimensions:
    :param max_len:
    :param layer_name:
    :param word_index_map:
    :param trainable:
    :return:
    """
    embeddings = load_embeddings(embeddings_file, embedding_dimensions=embeddings_dimensions)
    embeddings_matrix = create_embeddings_matrix(embeddings, word_index_map, embeddings_dimensions)
    return get_embeddings_layer(embeddings_matrix, name=layer_name, max_len=max_len,
                                trainable=trainable)
