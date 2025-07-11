# Copyright (c) Meta Platforms, Inc. and affiliates


import json
import os
import math

import sys
sys.path.append(".")

wiki_lang_code = [
    'ab', 'ace', 'ady', 'af', 'als', 'alt', 'am', 'ami', 'an', 'ang', 'anp', 'ar', 'arc', 'ary', 'arz', 'as', 'ast', 'atj', 'av', 'avk', 'awa', 'ay', 'az', 'azb', 
    'ba', 'ban', 'bar', 'bat_smg', 'bbc', 'bcl', 'be', 'be_tarask', 'bew', 'bg', 'bh', 'bi', 'bjn', 'blk', 'bm', 'bn', 'bo', 'bpy', 'br', 'bs', 'bug', 'bxr',
    'ca', 'cbk_zam', 'cdo', 'ce', 'ceb', 'ch', 'chr', 'chy', 'ckb', 'co', 'cr', 'crh', 'cs', 'csb', 'cu', 'cv', 'cy',
    'da', 'dag', 'de', 'dga', 'din', 'diq', 'dsb', 'dtp', 'dty', 'dv', 'dz', 
    'ee', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'ext',
    'fa', 'fat', 'ff', 'fi', 'fiu_vro', 'fj', 'fo', 'fon', 'fr', 'frp', 'frr', 'fur', 'fy',
    'ga', 'gag', 'gan', 'gcr', 'gd', 'gl', 'glk', 'gn', 'gom', 'gor', 'got', 'gpe', 'gu', 'guc', 'gur', 'guw', 'gv',
    'ha', 'hak', 'haw', 'he', 'hi', 'hif', 'hr', 'hsb', 'ht', 'hu', 'hy', 'hyw',
    'ia', 'id', 'ie', 'ig', 'igl', 'ik', 'ilo', 'inh', 'io', 'is', 'it', 'iu',
    'ja', 'jam', 'jbo', 'jv',
    'ka', 'kaa', 'kab', 'kbd', 'kbp', 'kcg', 'kg', 'ki', 'kk', 'kl', 'km', 'kn', 'ko', 'koi', 'krc', 'ks', 'ksh', 'ku', 'kus', 'kv', 'kw', 'ky', 
    'la', 'lad', 'lb', 'lbe', 'lez', 'lfn', 'lg', 'li', 'lij', 'lld', 'lmo', 'ln', 'lo', 'lt', 'ltg', 'lv',
    'mad', 'mai', 'map_bms', 'mdf', 'mg', 'mhr', 'mi', 'min', 'mk', 'ml', 'mn', 'mni', 'mnw', 'mr', 'mrj', 'ms', 'mt', 'mwl', 'my', 'myv', 'mzn',
    'nah', 'nap', 'nds', 'nds_nl', 'ne', 'new', 'nia', 'nl', 'nn', 'no', 'nov', 'nqo', 'nrm', 'nso', 'nv', 'ny',
    'oc', 'olo', 'om', 'or', 'os',
    'pa', 'pag', 'pam', 'pap', 'pcd', 'pcm', 'pdc', 'pfl', 'pi', 'pih', 'pl', 'pms', 'pnb', 'pnt', 'ps', 'pt', 'pwn',
    'qu',
    'rm', 'rmy', 'rn', 'ro', 'roa_rup', 'roa_tara', 'ru', 'rue', 'rw', 
    'sa', 'sah', 'sat', 'sc', 'scn', 'sco', 'sd', 'se', 'sg', 'sh', 'shi', 'shn', 'si', 'simple', 'sk', 'skr', 'sl', 'sm', 'smn', 'sn', 'so', 'sq', 'sr', 'srn', 'ss', 'st', 'stq', 'su', 'sv', 'sw', 'szl', 'szy',
    'ta', 'tay', 'tcy', 'te', 'tet', 'tg', 'th', 'ti', 'tk', 'tl', 'tly', 'tn', 'to', 'tpi', 'tr', 'trv', 'ts', 'tt', 'tum', 'tw', 'ty', 'tyv',
    'udm', 'ug', 'uk', 'ur', 'uz',
    've', 'vec', 'vep', 'vi', 'vls', 'vo',
    'wa', 'war', 'wo', 'wuu',
    'xal', 'xh', 'xmf', 
    'yi', 'yo', 'za', 'zea', 'zgh', 'zh', 'zh_classical', 'zh_min_nan', 'zh_yue', 'zu'
]


wordnet_to_wiki = {
    'fin': 'fi',  # Finnish
    'heb': 'he',  # Hebrew
    'slv': 'sl',  # Slovenian
    'ita': 'it',  # Italian
    'nno': 'nn',  # Norwegian Nynorsk
    'nob': 'no',  # Norwegian Bokmål
    'als': 'als',  # Alemannic
    'pol': 'pl',  # Polish
    'hrv': 'hr',  # Croatian
    'nld': 'nl',  # Dutch
    'ron': 'ro',  # Romanian
    'arb': 'ar',  # Arabic
    'isl': 'is',  # Icelandic
    'swe': 'sv',  # Swedish
    'por': 'pt',  # Portuguese
    'cmn': 'zh',  # Mandarin Chinese
    'jpn': 'ja',  # Japanese
    'dan': 'da',  # Danish
    'slk': 'sk',  # Slovak
    'lit': 'lt',  # Lithuanian
    'bul': 'bg',  # Bulgarian
    'eus': 'eu',  # Basque
    'cat': 'ca',  # Catalan
    'glg': 'gl',  # Galician
    'spa': 'es',  # Spanish
    'ell': 'el',  # Greek
    'zsm': 'ms',  # Malay
    'ind': 'id',  # Indonesian
    'fra': 'fr',  # French
    'tha': 'th',  # Thai
}

wiki_to_wordnet = {wiki_code: wordnet_code for wordnet_code, wiki_code in wordnet_to_wiki.items()}


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


ngram_dir = 'data/metadata_source/wiki_ngrams'
title_dir = "data/metadata_source/wiki_title"

wordnet_fn = 'data/metadata_source/wordnet_per_lang.json'
max_len = 256


out_dir = f"data/metadata_source/metadata_per_lang"


threshold_unigram = 0.1
threshold_bigram = 30.
upper_bound_unigram = 251_465
upper_bound_bigram = 100_646

fn_capped_unigram_num = lambda unigram_num, lang_code: min(upper_bound_unigram, int(unigram_num * threshold_unigram))
fn_capped_bigram_num = lambda bigram_num, lang_code: min(upper_bound_bigram, bigram_num)

threshold_title = 0.76
upper_bound_title = 61_235
fn_capped_title_num = lambda title_num: min(upper_bound_title, int(title_num * 0.76))


def build_metadata(lang_code):
    print(lang_code)
    if lang_code == 'en':  # skip and copy from MetaCLIPv1
        return

    if lang_code in wiki_to_wordnet:
        with open(wordnet_fn) as f:
            wordnet_per_lang = json.load(f)[wiki_to_wordnet[lang_code]]
    else:
        wordnet_per_lang = []
    
    wordnet_num = len(wordnet_per_lang)

    file_ngram = f'{ngram_dir}/{lang_code}.json'

    with open(file_ngram, 'r') as f:
        data_ngram = json.load(f)

    from pathlib import Path
    unigram_paths = [unigram_path for unigram_path in Path('data/wiki/').glob(f'{lang_code}wiki*.txt')]

    unigram_num = len(data_ngram["unigram"])
    bigram_num = len(data_ngram["bigram"])

    with open(f'{title_dir}/title_per_lang/{lang_code}.json') as f:
        title_per_lang = json.load(f)

    title_num = len(title_per_lang)

    print(f"[raw] {lang_code}: {wordnet_num} | {unigram_num} | {bigram_num} | {title_num} ")

    capped_unigram_num = fn_capped_unigram_num(unigram_num, lang_code)
    capped_bigram_num = fn_capped_bigram_num(bigram_num, lang_code)
    capped_title_num = fn_capped_title_num(title_num)
    
    total = wordnet_num + capped_unigram_num + capped_bigram_num + capped_title_num
    print(f"[cap] {lang_code}: {wordnet_num} | {capped_unigram_num} | {capped_bigram_num} | {capped_title_num} | {total}")

    metadata = {}
    
    # every lang has numbers.
    for m in range(100):
        m = str(m)
        if m not in metadata:
            metadata[m] = ''

    if wordnet_num:
        for m in wordnet_per_lang:
            if lang_code in ['zh', 'ja', 'fr'] and "+" in m:
                m = m.replace("+", "")
            m = m.strip()
            if len(m) == 0:
                continue
            if is_pure_punctuations(m):
                continue
            if m not in metadata:
                metadata[m] = 'w'
        merge_wordnet_num = len(metadata)
    else:
        merge_wordnet_num = 0

    
    data_list_unigram = sorted(data_ngram["unigram"].items(), key=lambda x: x[1], reverse=True)
    
    unigram_cnt = 0
    for m, _ in data_list_unigram:  # added here is a bug. dedup newly added cap.
        m = m.strip()
        if len(m) == 0 or len(m) > max_len:
            continue
        if is_pure_punctuations(m):
            continue
        if m not in metadata:
            metadata[m] = 'u'
            unigram_cnt += 1
            if unigram_cnt >= capped_unigram_num:
                break
    
    merge_unigram = len(metadata) - merge_wordnet_num

    is_not_split_by_space = lang_code in no_space_languages

    pmi_scores = {}

    for m, pmi in data_ngram["pmi"]:
        if pmi > threshold_bigram:
            pmi_scores[m] = pmi

    pmi_scores = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)

    bigram_cnt = 0
    for m, _ in pmi_scores:
        if is_not_split_by_space:
            m = "".join(m.split(" "))
        m = m.strip()
        if len(m) == 0 or len(m) > max_len:
            continue
        if is_pure_punctuations(m):
            continue
        if m not in metadata:
            metadata[m] = 'b'
            bigram_cnt += 1
            if bigram_cnt >= capped_bigram_num:
                break

    merge_bigram = len(metadata) - merge_unigram - merge_wordnet_num
    

    title_scores = sorted(title_per_lang.items(), key=lambda x: x[1], reverse=True)

    title_cnt = 0
    for m, _ in title_scores:
        m = m.strip()
        if len(m) == 0 or len(m) > max_len:
            continue
        if is_pure_punctuations(m):
            continue
        if m not in metadata:
            metadata[m] = 't'
            title_cnt += 1
            if title_cnt >= capped_title_num:
                break

    merge_title = len(metadata) - merge_bigram - merge_unigram - merge_wordnet_num
    
    from pathlib import Path
    Path(f'{out_dir}').mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/{lang_code}.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False)


if __name__ == '__main__':
    print(out_dir)
    import sys
    if len(sys.argv) == 2:
        build_metadata(sys.argv[1])
    else:
        import os
        import submitit
        import sys
        import math

        job_plans = [
            ("learn", len(wiki_lang_code)),
        ]

        for partition, array_size in job_plans:
            params = dict(
                name=f"build_metadata",
                gpus_per_node=0,
                mem_gb=80,
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
                for lang_code in wiki_lang_code:
                    job = executor.submit(
                        build_metadata,
                        lang_code
                    )
                    jobs.append(job)

            if len(jobs) > 0:
                print(partition, len(jobs), jobs[0].job_id, jobs[-1].job_id)
