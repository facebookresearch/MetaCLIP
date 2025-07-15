# Copyright (c) Meta Platforms, Inc. and affiliates

# Reference: https://www.nltk.org/howto/wordnet.html

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn
from collections import defaultdict
import json

entries = set()
for ss in wn.all_synsets():
    name = ss.name()
    dot_idx = name.find(".")
    name = name[:dot_idx].replace("_", " ")
    entries.add(name)

print(sorted(wn.langs()), f"Number of entries: {len(entries)}")
# print(sorted(wn.langs())) only English is available

# trigger to making multilingual synsets awake
_ = wn.synsets(b'\xe7\x8a\xac'.decode('utf-8'), lang='jpn')

multilingual_synsets = defaultdict(set)
for lang in wn.langs():
    for ss in wn.all_synsets(lang=lang):
        names = ss.lemma_names(lang)
        for name in names:
            name = name.replace("_", " ").replace("+", "")
            multilingual_synsets[lang].add(name)
    multilingual_synsets[lang] = sorted(list(multilingual_synsets[lang]))

print(f"Total number of synsets across all languages: {sum(len(synsets) for synsets in multilingual_synsets.values())}")

output_path = "data/metadata_source/wordnet_per_lang.json"

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(multilingual_synsets, f, ensure_ascii=False, indent=4)