#!/usr/bin/env bash

echo "Downloading English Wikipedia data."
echo "These files will be quite large (16 Gigs zipped, 66 Gigs unzipped) please ensure you have sufficient disk space."
echo "A dedicated data directory is recommended."
cd ../data
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2 .
bzip2 -d enwiki-latest-pages-articles-multistream.xml.bz2
mkdir -p english_wikipedia/json
mv enwiki-latest-pages-articles-multistream.xml ./english_wikipedia/
cd -
echo "Done!"