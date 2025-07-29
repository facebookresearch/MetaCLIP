# Building Metadata Entries for MetaCLIP

tl;dr:
```bash
python metaclip/build_metadata.py
```


## Part 1: WordNet synsets
`metaclip/build_metadata.py:wordnet_synsets`

```bash
pip install nltk
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4
```

## Part 2: Wiki Unigram
`metaclip/build_metadata.py:wiki_unigram`

Keep unigrams more than `100` occurences.

## Part 3: Wiki Bigrams
`metaclip/build_metadata.py:wiki_bigrams`

Computing pointwise mutual information more than 30.

## Part 4: Wiki Article Titles
`metaclip/build_metadata.py:wiki_title`

Keep view frequency more than `70`.
We randomly sample 25 days of [pageviews](https://dumps.wikimedia.org/other/pageviews) from past 5 years.
