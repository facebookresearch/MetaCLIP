# Copyright (c) Meta Platforms, Inc. and affiliates


import json
import pickle

from pathlib import Path
from collections import defaultdict

import sys
sys.path.append(".")

from metaclip.metadata.build_metadata import wiki_lang_code, is_punctuation

from metaclip.curation.substr_matching import lid_to_wiki, get_spaced_metadata_ml, initialize_automaton, substr_match, spacing


output_dir = f"data/metadata"
metadata_dir = f"data/metadata_source/metadata_per_lang"

print(metadata_dir, "->", output_dir)

# step1: merge and check wiki lang_code;
wiki_to_lid = {}
for lid_code, wiki_code in lid_to_wiki.items():
    if lid_to_wiki[lid_code] != "N/A":
        wiki_to_lid[wiki_code] = lid_code
    else:
        wiki_to_lid["other"] = "other"
print(f"wiki_to_lid={len(wiki_to_lid)}")

# step2: merge wiki lang_code;
merged_wiki_code = defaultdict(list)

for wiki_code in wiki_lang_code:
    if "_" in wiki_code:
        wiki_code_key = wiki_code.split("_")[0]
    elif wiki_code not in wiki_to_lid:
        wiki_code_key = "other"
    else:
        wiki_code_key = wiki_code

    merged_wiki_code[wiki_code_key].append(wiki_code)

print(f"merged_wiki_code={len(merged_wiki_code)}")

# step3: dump valid wiki metadata;
for wiki_code_key in merged_wiki_code:
    if wiki_code_key not in wiki_to_lid:
        continue

    if Path(f'{output_dir}/{wiki_code_key}.pkl').exists():
        continue

    print(wiki_code_key)
    metadata_set = set()
    for wiki_code in merged_wiki_code[wiki_code_key]:
        metadata_fn = f'{metadata_dir}/{wiki_code}.json'
        if not Path(metadata_fn).exists():
            print(metadata_fn, "does not exist.")
            if wiki_code != 'be_tarask':  # only skip one low res. language;
                raise ValueError(f"missing {metadata_fn} .")
            else:
                continue
        with open(metadata_fn) as f:
            for m, source in json.load(f).items():
                metadata_set.add(m)

    metadata = list(metadata_set)
    Path(f'{output_dir}').mkdir(parents=True, exist_ok=True)
    
    with open(f'{output_dir}/{wiki_code_key}.json', "w") as f:
        json.dump(metadata, f)
    
    spaced_metadata = get_spaced_metadata_ml(metadata)
    automaton = initialize_automaton(spaced_metadata)

    with open(f'{output_dir}/{wiki_code_key}.pkl', 'wb') as f:
        pickle.dump(automaton, f)
