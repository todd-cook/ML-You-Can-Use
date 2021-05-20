"""`install_corpora.py` - install free data corpora."""
import logging

from cltk.data.fetch import FetchCorpus

if __name__ == '__main__':

    LOG = logging.getLogger(__name__)
    LOG.addHandler(logging.NullHandler())
    logging.basicConfig(level=logging.INFO)
    try:
        corpus_importer = FetchCorpus('latin')
        corpus_importer.import_corpus('latin_text_latin_library')
        corpus_importer.import_corpus('latin_text_perseus')
        corpus_importer.import_corpus('latin_text_tesserae')
        corpus_importer = FetchCorpus('greek')
        corpus_importer.import_corpus('greek_text_perseus')
        corpus_importer.import_corpus('greek_text_lacus_curtius')
        logging.disable(logging.NOTSET)
    except:
        LOG.exception('Failure to download test corpora')
