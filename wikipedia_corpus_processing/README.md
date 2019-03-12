# Wikipedia Corpus Processing

## Preparation: Make sure your git pull included submodules, e.g.
    * ``git pull --recurse-submodules``  

## Latin wikipedia
* archive size: circa 86 MB, 410 MB unzipped
* after Jsonl processing: circa 79 MB
* corpus size: circa 36 MB

### To begin working with the Latin Wikipedia data, open a terminal and run:
 * run the `install_latin_wikipedia_data.sh` script
 * then run the `preprocess_latin_wikipedia_files.sh` script
  
## English Wikipedia
* archive size: circa 16 GB, unzipped 66 GB
* after Jsonl processing: circa 13 GB
* corpus size: 7 GB
 
### To begin working with the English Wikipedia data, open a terminal and run:
 * run the `install_english_wikipedia_data.sh` script
 * then run the `preprocess_english_wikipedia_files.sh` script

Now you are ready to run the notebooks in this directory.
