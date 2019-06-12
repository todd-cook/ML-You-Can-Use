"""`embeddings.py` - utility methods for working with embeddings."""

import os
import gzip
import logging
import mimetypes
import pickle
import shutil
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
from numpy import ndarray
from gensim import models
from keras.layers import Embedding
from keras.utils import get_file

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
        'http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip', 'glove.840B.300d.zip'],
    # Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 200d vectors, 1.42 GB download):
    # glove.twitter.27B.zip
    'Twitter2B': ['http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip',
                  'glove.twitter.27B.zip'],
    #     Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download):
    #     glove.42B.300d.zip
    'CommonCrawl.42B': ['http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip',
                        'glove.42B.300d.zip']
}  # type: Dict[str, List[str]]


def decompress(the_filepath: Path) -> None:
    """
    Decompress a gzip file, unless it's already decompressed.
    :param the_filepath:
    :return:
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


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

def get_embeddings_index(embedding_name: str,
                         url: str = None,
                         embeddings_filename: str = None,
                         parentdir: str = None,
                         cache_dir: str = None,
                         embedding_dimensions: int = 300) -> Dict[str, ndarray]:
    """
    High level function for get an embedding index, usually from a public url, downloading and
    caching locally.

    :param embedding_name: the name of the embeddings, used to look up metadata values
    :param url: the URL where the embeddings may be found; this parameter overrides the baked in
    metadata paths
    :param embeddings_filename: the filename; usually appended onto the url
    :param parentdir: where to store the files locally, if not specified then the keras cache
    directory will be used.
    :param cache_dir: where to store the files locally, if parentdir is not specified then the
    keras cache directory will be used.
    :param embedding_dimensions: integer: 300 or 100, 50 etc
    :return: a dictionary of strings to ndarray of embedding values
    """
    file_template = ''
    if embedding_name in EMBEDDINGS_METADATA:
        url, file_template = EMBEDDINGS_METADATA[embedding_name]

    if not cache_dir and parentdir:
        cache_dir = os.path.join(parentdir, 'data', embedding_name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    parts = urlparse(str(url))
    filename = parts.path.split('/')[-1]

    LOG.info('Downloading, please wait.')
    data_archive = get_file(
        fname=filename,
        origin=url,
        cache_dir=cache_dir,
        untar=False,
        extract=True)  # pylint disable:unused-variable
    LOG.info('Done downloading')

    if not data_archive:
        LOG.warning('Fail in fetch')

    if embeddings_filename:
        embeddings_file = embeddings_filename
    else:
        if '{' in file_template and '}' in file_template:
            embeddings_file = file_template.format(embedding_dimensions)

    parent_path = Path(str(cache_dir))
    embeddings_dir = parent_path / 'datasets'
    assert embeddings_dir.exists()

    embed_file = embeddings_dir / embeddings_file
    assert embed_file.exists()

    if str(embed_file).endswith('.gz'):
        decompress(embed_file)
        embed_file_str = str(embed_file)
        embed_file_str = embed_file_str[:embed_file_str.rfind('.')]
        embed_file = Path(embed_file_str)

    embeddings_index = load_embeddings(str(embed_file))
    return embeddings_index


def load_embeddings(embedding_file: str) -> Dict[str, ndarray]:
    """
    Low level function for loading embeddings from an accessible path.

    :param embedding_dir: valid directory
    :param embedding_file: valid file
    :return: a dictionary of strings to ndarray of embedding values
    """

    mime, _ = mimetypes.guess_type(embedding_file)
    embeddings_index = {}  # type: Dict[str, ndarray]
    if str(mime).startswith('text'):
        with open(embedding_file) as the_file:
            for line in the_file:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    else:
        try:
            keyed_vectors = models.KeyedVectors.load(embedding_file)
        except pickle.UnpicklingError:
            LOG.exception('failure using load')
            keyed_vectors = models.KeyedVectors.load_word2vec_format(embedding_file, binary=True,
                                                                     unicode_errors='replace')
        for word in keyed_vectors.vocab:
            embeddings_index[word] = keyed_vectors[word]
    LOG.info('Found %s word vectors.', len(embeddings_index))
    return embeddings_index


def create_embeddings_matrix(embeddings_index: Dict[str, ndarray],
                             vocabulary: Dict[int, str],
                             embeddings_dimensions: int = 100,
                             init_by_variance: bool = True) -> ndarray:
    """
    Create an embedding matrix

    :param embeddings_index: a dictionary mapping words to ndarrays
    :param vocabulary: a dictionary of words, matching unique incrementing integer indices to
    distinct words
    :param embeddings_dimension: the dimensions of the embeddings
    :param init_by_variance: boolean; initialize matrix to random values with each column
    initialized to the corresponding the variance of the seed embeddings columns.
    :return: an integer index ordered matrix of the filtered embedding values.
    """
    # pylint: disable=no-member
    if init_by_variance:
        cols = list(embeddings_index.values())[0].shape[0]
        embed_ar = np.asarray([tmp for tmp in embeddings_index.values()])
        matrix_variance = np.asarray(
            [np.var(embed_ar[:, idx]) for idx in range(cols)])
        del embed_ar
        embeddings_matrix = matrix_variance * np.random.rand(
            len(vocabulary) + 1, embeddings_dimensions)
    else:
        embeddings_matrix = np.random.rand(
            len(vocabulary) + 1, embeddings_dimensions)
    for i, word in enumerate(vocabulary):
        embedding_vector = embeddings_index.get(word)  # type: ignore
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
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
                            int_word_map: Dict[int, str],
                            trainable: bool = True):
    """
    Simplified helper method
    :param embeddings_file:
    :param embeddings_dimensions:
    :param max_len:
    :param layer_name:
    :param int_word_map:
    :param trainable:
    :return:
    """
    embeddings = load_embeddings(embeddings_file)
    embeddings_matrix = create_embeddings_matrix(embeddings, int_word_map, embeddings_dimensions)
    return get_embeddings_layer(embeddings_matrix, name=layer_name, max_len=max_len,
                                trainable=trainable)
