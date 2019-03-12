#!/usr/bin/env bash

echo "Installing Python Certificates"
/Applications/Python\ 3.7/Install\ Certificates.command || echo "Certificate install fail; let's hope you don't need them."

echo "Installing CLTK Data"
python install_corpora.py

echo "Installing NLTK data"
python -m nltk.downloader all

echo "Downloading Wikipedia data"
cd data
wget https://dumps.wikimedia.org/lawiki/latest/lawiki-latest-pages-articles-multistream.xml.bz2 .
bzip2 -d lawiki-latest-pages-articles-multistream.xml.bz2
mkdir -p latin_wikipedia/json
mv lawiki-latest-pages-articles-multistream.xml ./latin_wikipedia/

#wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2 .
#bzip2 -d enwiki-latest-pages-articles-multistream.xml.bz2
#mkdir -p english_wikipedia/json
#mv enwiki-latest-pages-articles-multistream.xml ./english_wikipedia/
cd ..

echo "Done!"