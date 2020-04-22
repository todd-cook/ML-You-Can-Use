Labeling Data
===========

## Sequence of notebooks
* [labeling_occupation_data_with_Wikipedia_and_GoogleNews](labeling_occupation_data_with_Wikipedia_and_GoogleNews.ipynb)
    * data in: `occupations.wikidata.csv`
    * data out: `occupations.wikidata.all.gnews.labeled.csv`

* [correcting_GoogleNews_labels_with_Cleanlab](correcting_GoogleNews_labels_with_Cleanlab.ipynb)
    * data in: `occupations.wikidata.all.gnews.labeled.csv`
    * data out: `occupations.wikidata.all.gnews.labeled.final.csv`
    
* [training_to_label_with_BERT_and_Cleanlab](training_to_label_with_BERT_and_Cleanlab.ipynb)
    * data in: `occupations.wikidata.all.gnews.labeled.final.csv`
    * data out: `occupations.wikidata.all.labeled.csv`

Labeling Guidelines:

Occupation: an activity in which one engages; vocation (m-w.com dictionary definition)
Most commonly, an occupation should indicate a person doing a job. 
Used in a sentence, a person could be substituted for the occupation.

* I work in finance.
* I work as an accountant.

Saying "finance" is an occupation is substituting the higher level domain for the occupation.
Similarly:

* hair - no
* barber - yes
* stylist - yes
 
* oncology - no, that's the field
* oncologist - yes, that's the person specializing in the field

etc.
 
Plurals are invalid because they indicate an abstract group of people, not a single person performing the work:

* teachers - no
* teacher - yes
 
* software - no
* software engineer - yes
* principal engineer - yes (even though it blends Position/title with Occupation)

So, some flexibility in labeling is desirable; it's important to screen out the extreme bad cases.

