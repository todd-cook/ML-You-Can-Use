#!/usr/bin/env bash

echo "Make sure your have run: git pull --recurse-submodules"
echo "Preparing Json from wikimedia dumps"
cd ../data/wikiextractor
  ./WikiExtractor.py ../enwiki-latest-pages-articles-multistream.xml -o ../english_wikipedia/jsonl --json
cd ../../wikipedia_corpus_processing/
