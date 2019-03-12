#!/usr/bin/env bash

echo "Downloading Latin Wikipedia data"
cd ../data
wget https://dumps.wikimedia.org/lawiki/latest/lawiki-latest-pages-articles-multistream.xml.bz2 .
bzip2 -d lawiki-latest-pages-articles-multistream.xml.bz2
mkdir -p latin_wikipedia/json
mv lawiki-latest-pages-articles-multistream.xml ./latin_wikipedia/
cd -
echo "Done!"