"""`install_corpora.py` - install free data corpora."""
import logging

from cltk.corpus.utils.importer import CorpusImporter

if __name__ == '__main__':

    LOG = logging.getLogger(__name__)
    LOG.addHandler(logging.NullHandler())
    logging.basicConfig(level=logging.INFO)

    try:
        corpus_importer = CorpusImporter('latin')
        corpus_importer.import_corpus('latin_text_latin_library')
        corpus_importer.import_corpus('latin_text_perseus')
        corpus_importer.import_corpus('latin_text_tesserae')
        corpus_importer = CorpusImporter('greek')
        corpus_importer.import_corpus('greek_text_perseus')
        corpus_importer.import_corpus('greek_text_lacus_curtius')
    except:
        LOG.exception('Failure to download test corpora')
