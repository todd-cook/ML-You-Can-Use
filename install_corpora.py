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
        from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
        bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        bert_model = BertForMaskedLM.from_pretrained("bert-large-cased-whole-word-masking")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        logging.disable(logging.NOTSET)
    except:
        LOG.exception('Failure to download test corpora')
