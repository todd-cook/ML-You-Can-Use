#!/usr/bin/env bash

echo "Make sure your have run: git pull --recurse-submodules"
echo "Preparing Json from wikipedia dumps"
cd ../data/wikiextractor
 ./WikiExtractor.py ../lawiki-latest-pages-articles-multistream.xml -o ../latin_wikipedia/jsonl --json
cd -
echo "Done"
