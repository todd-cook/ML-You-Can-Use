# ML-You-Can-Use
[![CircleCI](https://circleci.com/gh/todd-cook/ML-You-Can-Use.svg?style=svg)](https://circleci.com/gh/todd-cook/ML-You-Can-Use)  [![codecov.io](http://codecov.io/github/todd-cook/ML-You-Can-Use/coverage.svg?branch=master)](http://codecov.io/github/todd-cook/ML-You-Can-Use?branch=master)

Practical Machine Learning and Natural Language Processing with examples.

## Featuring
* Interesting applications of NLP and ML.
* Practical demonstration notebooks
* Reproducible experiments
* Illustrated best practices:
    * Code extracted from notebooks for:
        * Type checking via MyPy annotations
        * Linting via Pylint
        * Doctests whenever possible


### Setup
Download this repo using git with the submodule command, e.g.:

``git pull --recurse-submodules``

Submodules are used to pull in some data and external data processing utilities that we'll use for preprocessing some of the data.

### Install Python 3
### Create Virtual Environment
``` 
mkdir p3
 `which python3` -m venv ./p3
 source p3/bin/activate
```
### Install Requirements

``pip install -r requirements.txt``

### Installing Test Corpora

``install_corpora.sh``

* installs Python ssl certificates
* installs CLTK data for Latin and Greek
* installs NLTK data
* downloads Wikipedia data dump for local processing

### Testing
``./runUnitTests.sh``

### Interactivity
``juypter notebook`` 

## Recommended Sequence

### Building a Language Model
* [1. Assessing Corpus Quality](building_language_model/assessing_corpus_quality.ipynb)
* [2. Making a Frequency Distribution](building_language_model/make_frequency_distribution.ipynb)
* [3. Making a Trie Language Model](building_language_model/make_trie_language_model.ipynb)
### Detecting Loanwords
* [5. Making a Frequency Distribution of Transliterated Greek](detecting_loanwords/make_frequency_distribution_greek_transliterated.ipynb)
* [6. Boosting Training Data](detecting_loanwords/boosting_training_data.ipynb)
* [7. The Problem of Loanwords, and a Solution](detecting_loanwords/loanwords_problems_solutions.ipynb)
### Customized Word Embeddings
* [8. Quality Embeddings](quality_embeddings/generate_word_vector.ipynb) 
* [9. Wikipedia Corpus Processing](wikipedia_corpus_processing/create_corpus_from_latin_wiki.ipynb) 
