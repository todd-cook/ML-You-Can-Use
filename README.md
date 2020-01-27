# ML-You-Can-Use
[![CircleCI](https://circleci.com/gh/todd-cook/ML-You-Can-Use.svg?style=svg)](https://circleci.com/gh/todd-cook/ML-You-Can-Use)  [![codecov.io](http://codecov.io/github/todd-cook/ML-You-Can-Use/coverage.svg?branch=master)](http://codecov.io/github/todd-cook/ML-You-Can-Use?branch=master)

Practical Machine Learning and Natural Language Processing with examples.

## Featuring
* Interesting applications ML, NLP, Computer Vision
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
 source setPythonHashSeed.sh
 source p3/bin/activate
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

### Modeling Language
* [1. Assessing Corpus Quality](probablistic_language_modeling/assessing_corpus_quality.ipynb)
* [2. Making a Frequency Distribution](probablistic_language_modeling/make_frequency_distribution.ipynb)
* [3. Making a Word Trie Probability Model](probablistic_language_modeling/make_trie_language_model.ipynb)
### Detecting Duplicate Documents
* [4. Merge corpora by detecting and filtering duplicate documents](document_deduplication/find_duplicate_docs.ipynb)
### Classifying Texts
* [5. Benchmarking our classifier](text_classification/imdb_benchmark.ipynb)
* [6. Boostrapping Document Classification](text_classification/bootstrapping_document_classification.ipynb)
### Detecting Loanwords
* [7. Making a Frequency Distribution of Transliterated Greek](detecting_loanwords/make_frequency_distribution_greek_transliterated.ipynb)
* [8. Boosting Training Data](detecting_loanwords/boosting_training_data.ipynb)
* [9. The Problem of Loanwords, and a Solution](detecting_loanwords/loanwords_problems_solutions.ipynb)
* [10. Feature Engineering with the Loanwords matrix](detecting_loanwords/loanwords_feature_engineering.ipynb)
* [11. Detecting Loanwords with Keras](detecting_loanwords/detecting_loanwords_keras..ipynb)
### Wikipedia Corpus Processing
* [12. English Wikipedia Corpus Cleaning](wikipedia_corpus_processing/clean_english_wiki_corpus.ipynb)
* [13. English Wikipedia Corpus Processing](wikipedia_corpus_processing/create_corpus_from_english_wiki.ipynb)
* [14. Latin Corpus Processing](wikipedia_corpus_processing/create_corpus_from_latin_wiki.ipynb)
* [15. Down sample or not](wikipedia_corpus_processing/down_sample_or_not.ipynb)
### Quality Embeddings 
* [16. Generate English Wikipedia word vector](quality_embeddings/generate_latin_word_vector.ipynb) 
* [17. Generate Latin word vector](quality_embeddings/generate_latin_word_vector.ipynb) 
### Computer Vision - Object Detection
* [18. Object detection as a multivariable regression using a custom Convnet](computer_vision_object_detection/train_noisy_circle_detector.ipynb)
* [19. Assessing the Noisy Circle detector](computer_vision_object_detection/assess_noisy_circle_detector.ipynb)
### Summarizing Texts
* [20. Assessing Headline Generation](summarization/neural_headline_generation_metrics.ipynb)

# References and Acknowledgements    
* [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification by Ye Zhang, Byron Wallace](https://arxiv.org/abs/1510.03820)
* [Word2vec applied to Recommendation: Hyperparameters Matter byHugo Caselles-Dupré, Florian Lesaint, Jimena Royo-Letelier](https://arxiv.org/pdf/1804.04212)
* [Exploiting Similarities among Languages for Machine Translation by Tomas Mikolov, Quoc V. Le, Ilya Sutskever](https://arxiv.org/abs/1309.4168)
* [Distributed Representations of Words and Phrases and their Compositionality by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean ](https://arxiv.org/abs/1310.4546)
* [Deep Learning with Python by Francois Chollet](https://github.com/fchollet/deep-learning-with-python-notebooks)
* [Mining Massive Datasets](http://www.mmds.org)
* [Chris McCormick MinHash Tutorial with Python Code](http://mccormickml.com/2015/06/12/minhash-tutorial-with-python-code)
* [Convolutional Neural Networks for Text Classification by David S. Batista](http://www.davidsbatista.net/blog/2018/03/31/SentenceClassificationConvNets/)
* [Convolutional Neural Networks for Sentence Classification by Yoon Kim](https://arxiv.org/abs/1408.5882)

