# Copyright (c) Meta Platforms, Inc. and affiliates


import string
import os
import re
import json
import sys
import tqdm

from collections import defaultdict
from pathlib import Path


def is_cjk_or_similar(char):
    code_point = ord(char)
    # CJK ranges
    if (
        0x4E00 <= code_point <= 0x9FFF or
        0x3400 <= code_point <= 0x4DBF or
        0x20000 <= code_point <= 0x2A6DF or
        0x2A700 <= code_point <= 0x2B73F or
        0x2B740 <= code_point <= 0x2B81F or
        0x2B820 <= code_point <= 0x2CEAF or
        0x2CEB0 <= code_point <= 0x2EBEF or
        0xF900 <= code_point <= 0xFAFF or
        0x2E80 <= code_point <= 0x2EFF or
        0x2F00 <= code_point <= 0x2FDF or
        0x2FF0 <= code_point <= 0x2FFF
    ):
        return True
    # Thai, Lao, Burmese, Khmer, Tibetan ranges
    elif (
        0x0E00 <= code_point <= 0x0E7F or  # Thai
        0x0E80 <= code_point <= 0x0EFF or  # Lao
        0x1000 <= code_point <= 0x109F or  # Burmese
        0x1780 <= code_point <= 0x17FF or  # Khmer
        0x0F00 <= code_point <= 0x0FFF     # Tibetan
    ):
        return True
    # Punctuation
    elif is_punctuation(char):
        return True
    else:
        return False


def remove_html_tags(text):
    cleaned_text = re.sub(r'(&lt;.*?&gt;|<.*?>)', '', text)
    return cleaned_text


additional_punctuation = {k:0 for k in [
    '，', '。', '、', '；', '：', '？', '¿', '！', '“', '”', '‘', '’', '（', '）', '【', '】', '《', '》', '〈', '〉', '「', '」', '『', '』', '～', '—'
]}


def is_punctuation(char):
    return char in string.punctuation or char in additional_punctuation


def is_pure_punctuations(s):
    for c in s:
        if not is_punctuation(c):
            return False
    return True


def check_weird_token(token):
    paired_punc = [
        ["(", ")"],
        ["[", "]"],
        ["{", "}"],
        ["“", "”"],
        ["‘", "’"],
        ["（", "）"],
        ["【", "】"],
        ["《", "》"],
        ["〈", "〉"],
        ["「", "」"],
        ["『 ", "』"],
    ]

    if is_punctuation(token[0]):
        return True
    for pair in paired_punc:
        if (pair[0] in token and pair[1] not in token) or (pair[1] in token and pair[0] not in token):
            return True
    return False


no_space_languages = {
    'bo',
    'dz',
    'ja',
    'km',
    'lo',
    'my',
    'ryu',
    'th',
    'zh',
    'zh_classical',
    'zh_yue',
}


def simple_tokenizer(text):
    tokens = []
    current_token = []

    for char in text:
        if is_cjk_or_similar(char):
            if current_token:
                tokens.append(''.join(current_token))
                current_token = []
            tokens.append(char)
        elif char.isspace():
            if current_token:
                tokens.append(''.join(current_token))
                current_token = []
        else:
            current_token.append(char)

    if current_token:
        tokens.append(''.join(current_token))

    return tokens


def load_tokenizer(lang_code):
    if lang_code in ['bo', 'dz']:
        from metaclip.metadata.lang_tokenizers import bo
        return bo
    elif lang_code in ['ja', 'ryu']:
        from metaclip.metadata.lang_tokenizers import ja
        return ja
    elif lang_code == 'km':
        from metaclip.metadata.lang_tokenizers import km
        return km
    elif lang_code == 'lo':
        from metaclip.metadata.lang_tokenizers import lo
        return lo
    elif lang_code == 'my':
        from metaclip.metadata.lang_tokenizers import my
        return my
    elif lang_code == 'th':
        from metaclip.metadata.lang_tokenizers import th
        return th
    elif lang_code in ['zh', 'zh_classical', 'zh_yue']:
        from metaclip.metadata.lang_tokenizers import zh
        return zh
    else:
        return None


def special_language_tokenizer(text, lang_code):
    if type(text) == str:
        space_splitted_texts = text.split()
    else:
        space_splitted_texts = text

    tokenizer = load_tokenizer(lang_code)
    tokenized_space_splitted_texts = tokenizer.tokenize(space_splitted_texts)
    return tokenized_space_splitted_texts


def tokenize(text, lang_code):
    if lang_code not in no_space_languages:
        return simple_tokenizer(text)
    else:
        return [token for splitted_text in special_language_tokenizer(text.split(), lang_code) for token in splitted_text]


def count_ngrams(tokens, ngram_counts, lang_code):
    prev_token = None
    for token in tokens:
        token = token.strip()
        if len(token) == 0:
            continue

        if is_pure_punctuations(token):
            prev_token = None  # skip punctuation for bigram
            continue

        if check_weird_token(token) and lang_code in no_space_languages:
            prev_token = None
            continue

        ngram_counts["unigram"][token] += 1
        
        if prev_token is not None:
            ngram_counts["bigram"][f"{prev_token} {token}"] += 1

        prev_token = token


def build_ngrams(file_paths):
    for file_path in tqdm.tqdm(file_paths):
        lang_code, segment, fn = file_path.split("/")[-3:]
        lang_code = lang_code.split("_text")[0]

        ngram_fn = f"{wiki_ngram_path}/{lang_code}/{segment}/{fn}.json"
        if Path(ngram_fn).exists():
            continue

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        text = remove_html_tags(text)
        tokens = tokenize(text, lang_code)
        
        ngram_counts = {"unigram": defaultdict(int), "bigram": defaultdict(int)}
        count_ngrams(tokens, ngram_counts, lang_code)

        Path(f"{wiki_ngram_path}/{lang_code}/{segment}").mkdir(parents=True, exist_ok=True)
        with open(ngram_fn, "w") as fw:
            json.dump(ngram_counts, fw, ensure_ascii=False)


wiki_text_path = "data/metadata_source/wiki_text"
wiki_ngram_path = "data/metadata_source/wiki_ngram"


if __name__ == '__main__':
    rand_list_fn = f"rand_fns.json"  # random file list to balance processing time.
    if not Path(rand_list_fn).exists():
        import glob
        file_paths = glob.glob(os.path.join(wiki_text_path, '**', 'wiki_*'), recursive=True)
        import random
        random.seed(0)
        random.shuffle(file_paths)
        with open(rand_list_fn, "w") as fw:
            json.dump(file_paths, fw)
    else:
        with open(rand_list_fn) as f:
            file_paths = json.load(f)

    import sys
    if len(sys.argv) == 1:  # testing
        build_ngrams(file_paths)
    else:
        import os
        import submitit
        import sys
        import math

        job_plans = [	
            ("data", 512, 0, len(file_paths))
        ]

        for partition, array_size, start_shard, end_shard in job_plans:
            params = dict(
                name=f"build_ngram",
                gpus_per_node=1,
                mem_gb=40,
                cpus_per_task=4,
                nodes=1,
                slurm_partition=partition,
                timeout_min=4320,
            )

            executor = submitit.AutoExecutor(
                folder="submitit/%j"
            )
            executor.update_parameters(**params)

            jobs = []
            with executor.batch():
                shards_per_thread = math.ceil((end_shard - start_shard) / array_size)
                starts = list(range(start_shard, end_shard, shards_per_thread))
                ends = list(range(start_shard + shards_per_thread, end_shard + shards_per_thread, shards_per_thread))
                for start_shard, end_shard in zip(starts, ends):
                    job = executor.submit(
                        build_ngrams,
                        file_paths[start_shard:end_shard]
                    )
                    jobs.append(job)

            if len(jobs) > 0:
                print(partition, len(jobs), jobs[0].job_id, jobs[-1].job_id)
