#!/usr/bin/env bash

echo "Installing Python Certificates"
/Applications/Python\ 3.7/Install\ Certificates.command || echo "Certificate install fail; let's hope you don't need them."

echo "Installing CLTK Data"
python install_corpora.py

echo "Installing NLTK data"
python -m nltk.downloader all