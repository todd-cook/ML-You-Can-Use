# ML-You-Can-Use
Practical ML with examples. Support for articles, arguments, and tutorials.

## Featuring
* Interesting applications of Natural Language Processing and Machine Learning.
* Practical demonstration notebooks
* Longer methods and code extracted from the notebooks for type checking, linting, testing and increased robustness.
* Type checking via MyPy annotations
* Linting via Pylint
* Doctests whenever possible
* Pytest configuration

### Setup
### Install Python 3
### Create Virtual Environment
``` 
mkdir p37
 `which python3` -m venv ./p37
 source p37/bin/activate
```
### Install Requirements

``pip install -r requirements.txt``

### Installing Test Corpora

``install_corpora.sh``

* installs Python ssl certificates
* installs CLTK data for Latin and Greek
* installs NLTK data

### Testing
``./runUnitTests.sh``

### Interactivity
``juypter notebook`` 

## Recommended Sequence

### Building a Language Model
* [1. Assessing Corpus Quality](building_language_model/assessing_corpus_quality.ipynb)
* [2. Making a Frequency Distribution](building_language_model/make_frequency_distribution.ipynb)
* [3. Making a Trie Language Model](building_language_model/make_trie_language_model.ipynb)
### Bootstrapping Classification
* [4. Bootstrapping - Classifying Corpus Documents](bootstrapping_classification/corpus_doc_classify.ipynb)
### Detecting Loanwords
* [5. Making a Frequency Distribution of Transliterated Greek](detecting_loanwords/make_frequency_distribution_greek_transliterated.ipynb)
* [6. Boosting Training Data](detecting_loanwords/boosting_training_data.ipynb)
* [7. The Problem of Loanwords, and a Solution](detecting_loanwords/loanwords_problems_solutions.ipynb)
