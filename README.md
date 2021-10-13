# ML-You-Can-Use
[![Build Status](https://travis-ci.com/todd-cook/ML-You-Can-Use.svg?branch=master)](https://travis-ci.com/todd-cook/ML-You-Can-Use) [![CircleCI](https://circleci.com/gh/todd-cook/ML-You-Can-Use.svg?style=svg)](https://circleci.com/gh/todd-cook/ML-You-Can-Use) [![codecov.io](http://codecov.io/github/todd-cook/ML-You-Can-Use/coverage.svg?branch=master)](http://codecov.io/github/todd-cook/ML-You-Can-Use?branch=master)

Practical Machine Learning and Natural Language Processing with examples.

## Featuring
* Interesting applications of ML, NLP, and Computer Vision
* Practical demonstration notebooks
* Reproducible experiments
* Illustrated best practices:
    * Code extracted from notebooks for:
        * automatic formatting with Black
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
 source setPythonHashSeed.sh
 source p3/bin/activate
```
### Install Requirements

``pip install -r requirements.txt``

#### For running all notebook examples
``pip install -r requirements-dev.txt``

##### Note: some examples will have a conda `environment.yaml` file that you will want to use.

### Installing Test Corpora

Many notebooks use data that needs to be installed, do so by running the install script.

``install_corpora.sh``

* installs Python ssl certificates
* installs CLTK data for Latin and Greek
* installs NLTK data

### Testing
``./runUnitTests.sh``

### Interactivity
``juypter notebook`` 

## Notebooks

### Getting data
* [Extracting Occupation and Employer data from Wikidata](getting_data/extracting_occupation_and_employer_data_from_wikidata.ipynb) 

### Labeling Data
* [Labeling occupation data with Wikipedia and GoogleNews](labeling_data/labeling_occupation_data_with_Wikipedia_and_GoogleNews.ipynb)
* [Correcting GoogleNews labels with Cleanlab](labeling_data/correcting_GoogleNews_labels_with_Cleanlab.ipynb)
* [Training to label with BERT and Cleanlab](labeling_data/training_to_label_with_BERT_and_Cleanlab.ipynb)

### Modeling Language
* [Assessing Corpus Quality](probabilistic_language_modeling/assessing_corpus_quality.ipynb)
* [Making a Frequency Distribution](probabilistic_language_modeling/make_frequency_distribution.ipynb)
* [Making a Word Trie Probability Model](probabilistic_language_modeling/make_trie_language_model.ipynb)
* [Word and Sentence Probability using BERT](probabilistic_language_modeling/automatic_grammatical_error_corrections_using_BERT_GPT2.ipynb)
### Detecting Duplicate Documents
* [Merge corpora by detecting and filtering duplicate documents](document_deduplication/find_duplicate_docs.ipynb)
### Classifying Texts
* [Benchmarking our classifier](text_classification/imdb_benchmark.ipynb)
* [Boostrapping Document Classification](text_classification/bootstrapping_document_classification.ipynb)
### Detecting Loanwords
* [Making a Frequency Distribution of Transliterated Greek](detecting_loanwords/make_frequency_distribution_greek_transliterated.ipynb)
* [Boosting Training Data](detecting_loanwords/boosting_training_data.ipynb)
* [The Problem of Loanwords, and a Solution](detecting_loanwords/loanwords_problems_solutions.ipynb)
* [Feature Engineering with the Loanwords matrix](detecting_loanwords/loanwords_feature_engineering.ipynb)
* [Detecting Loanwords with Keras](detecting_loanwords/detecting_loanwords_keras.ipynb)
### Wikipedia Corpus Processing
* [English Wikipedia Corpus Cleaning](wikipedia_corpus_processing/clean_english_wiki_corpus.ipynb)
* [English Wikipedia Corpus Processing](wikipedia_corpus_processing/create_corpus_from_english_wiki.ipynb)
* [Latin Corpus Processing](wikipedia_corpus_processing/create_corpus_from_latin_wiki.ipynb)
* [Downsample or not](wikipedia_corpus_processing/down_sample_or_not.ipynb)
### Quality Embeddings 
* [Generating an English Wikipedia word vector](quality_embeddings/generate_latin_word_vector.ipynb) 
* [Generating a Latin word vector](quality_embeddings/generate_latin_word_vector.ipynb)
* [The Case for Using an Embedding Encoder](quality_embeddings/embedding_encoder.ipynb) 
* [Sentence Embeddings - A simple but effective baseline - using Seneca](quality_embeddings/sentence_similarities_seneca.ipynb)
### Computer Vision - Object Detection
* [Object detection as a multivariable regression using a custom Convnet](computer_vision_object_detection/train_noisy_circle_detector.ipynb)
* [Assessing the Noisy Circle detector](computer_vision_object_detection/assess_noisy_circle_detector.ipynb)
### Summarizing Texts
* [Assessing Headline Generation](summarization/neural_headline_generation_metrics.ipynb)
### Searching and Search Relevance
* [Search Results Relevance using BERT](searching/search_results_relevance_using_BERT.ipynb)

#### References and Acknowledgements    
* [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification by Ye Zhang, Byron Wallace](https://arxiv.org/abs/1510.03820)
* [Word2vec applied to Recommendation: Hyperparameters Matter byHugo Caselles-Dupré, Florian Lesaint, Jimena Royo-Letelier](https://arxiv.org/pdf/1804.04212)
* [Exploiting Similarities among Languages for Machine Translation by Tomas Mikolov, Quoc V. Le, Ilya Sutskever](https://arxiv.org/abs/1309.4168)
* [Distributed Representations of Words and Phrases and their Compositionality by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean ](https://arxiv.org/abs/1310.4546)
* [Deep Learning with Python by Francois Chollet](https://github.com/fchollet/deep-learning-with-python-notebooks)
* [Mining Massive Datasets](http://www.mmds.org)
* [Chris McCormick MinHash Tutorial with Python Code](http://mccormickml.com/2015/06/12/minhash-tutorial-with-python-code)
* [Convolutional Neural Networks for Text Classification by David S. Batista](http://www.davidsbatista.net/blog/2018/03/31/SentenceClassificationConvNets/)
* [Convolutional Neural Networks for Sentence Classification by Yoon Kim](https://arxiv.org/abs/1408.5882)
* [The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction by Dimitris Alikaniotis, Vipul Raheja](https://arxiv.org/abs/1906.01733)    
* [BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model by Alex Wang, Kyunghyun Cho](https://arxiv.org/pdf/1902.04094.pdf)
* [An Exploration of Word Embedding Initialization in Deep-Learning Tasks by Tom Kocmi, Ondřej Bojar](https://arxiv.org/pdf/1711.09160.pdf)