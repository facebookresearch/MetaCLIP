# Worldwide MetaCLIP 

## TODO
merge English curation and multilingual codebase.

## Metadata

`metaclip/metadata` contains code and steps to process metadata for each language. It requires parallel processing to handle 330 languages.

## Build Multilingual WordNet

(TODO)
assuming WordNet is dumped to `data/metadata_source/wordnet_per_lang.json`.


### Build Wikipedia ngrams

Download wikipedia corpora via `bash metaclip/metadata/download_wikipedia.sh <lang_code> data/metadata_source/wiki_text`.

`wikiextractor` is required to extract wiki text.

All wiki language code can be found in `metaclip/metadata/build_metadata.py`.

`python metaclip/metadata/build_ngram.py <submitit>`
(if special tokenizer is used, please install pkgs in `metaclip/metadata/requirements.txt` first.

### Build Wikipedia Title

`python metaclip/metadata/build_title.py <submitit>` for parallel processing of titles ranging in multiple data.
then `python metaclip/metadata/build_title.py` will merge processed titles from multiple dates.


### Put Together

`python metaclip/metadata/build_metadata.py` will merge metadata from 4 sources.

### Align Language ID and Build Automaton (faster substr matching)

`python metaclip/metadata/align_lid_to_wiki.py` will merge multiple wiki metadata to ensure one-to-one mapping from LID's language to a unique metadata.


## Curation

`metaclip/curation` contains code for Algorithm 1 and helper functions.

`python metaclip/curation/curate.py count_per_shard` to count matches for each shard of data.
`python metaclip/curation/curare.py` run global count.
`python metaclip/curation/curate.py curate` run curation/balancing for each example.
