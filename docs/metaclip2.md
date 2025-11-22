# MetaCLIP 2 Curation

Detailed step-by-step guide for building metadata and curating data for MetaCLIP's worldwide (multilingual) implementation, covering 329 languages.

## Overview

MetaCLIP Worldwide extends the original English-only MetaCLIP approach to support global languages for building worldwide datasets. The process is significantly more complex than the English version due to:

- Support metadata collected for 329 languages from Wikipedia
- Language-specific tokenization: Special handling for non-space-separated languages
- Cross-language metadata alignment with Language ID (LID) integration
- Language-dependent metadata thresholding/balancing/curation

## Prerequisites

### Dependencies
```bash
# Core dependencies
pip install nltk wikiextractor submitit==1.2.1

# Language-specific tokenizers (install as needed)
pip install -r metaclip/metadata/requirements.txt
```

### Directory Structure Setup
```bash
mkdir -p data/metadata_source/{wiki_text,wiki_ngrams,wiki_title,wordnet_per_lang}
mkdir -p data/metadata
mkdir -p data/index_json
mkdir -p data/curated_index_json
mkdir -p data/pkg_json
mkdir -p data/valid_uuids
```

## Phase 1: Metadata Source Building

### Step 1: Download Wikipedia Corpora

Download Wikipedia text for all 329 supported languages:

```bash
# Download all Wikipedia corpora (parallel processing recommended rather than sequential)
for lang_code in ab ace ady af als alt am ami an ang anp ar arc ary arz as ast atj av avk awa ay az azb ba ban bar bat_smg bbc bcl be be_tarask bew bg bh bi bjn blk bm bn bo bpy br bs bug bxr ca cbk_zam cdo ce ceb ch chr chy ckb co cr crh cs csb cu cv cy da dag de dga din diq dsb dtp dty dv dz ee el eml en eo es et eu ext fa fat ff fi fiu_vro fj fo fon fr frp frr fur fy ga gag gan gcr gd gl glk gn gom gor got gpe gu guc gur guw gv ha hak haw he hi hif hr hsb ht hu hy hyw ia id ie ig igl ik ilo inh io is it iu ja jam jbo jv ka kaa kab kbd kbp kcg kg ki kk kl km kn ko koi krc ks ksh ku kus kv kw ky la lad lb lbe lez lfn lg li lij lld lmo ln lo lt ltg lv mad mai map_bms mdf mg mhr mi min mk ml mn mni mnw mr mrj ms mt mwl my myv mzn nah nap nds nds_nl ne new nia nl nn no nov nqo nrm nso nv ny oc olo om or os pa pag pam pap pcd pcm pdc pfl pi pih pl pms pnb pnt ps pt pwn qu rm rmy rn ro roa_rup roa_tara ru rue rw sa sah sat sc scn sco sd se sg sh shi shn si simple sk skr sl sm smn sn so sq sr srn ss st stq su sv sw szl szy ta tay tcy te tet tg th ti tk tl tly tn to tpi tr trv ts tt tum tw ty tyv udm ug uk ur uz ve vec vep vi vls vo wa war wo wuu xal xh xmf yi yo za zea zgh zh zh_classical zh_min_nan zh_yue zu; do
    bash metaclip/metadata/download_wikipedia.sh $lang_code data/metadata_source/wiki_text
done
```

**Note**: This step requires significant storage (~100GB+) and bandwidth. Consider using a cluster or cloud environment.

### Step 2: Build Multilingual WordNet

Extract and organize Multilingual WordNet synsets for supported languages:

```bash
python metaclip/metadata/build_multilingual_wordnet.py
```

This creates `data/metadata_source/wordnet_per_lang.json` with Multilingual WordNet synsets mapped to language codes.

**Language Mapping**: The system maps between WordNet language codes and Wikipedia language codes using the mapping in `metaclip/metadata/build_metadata.py:40-71`.

### Step 3: Build Wikipedia N-grams

Generate unigrams and bigrams with frequency and PMI scores for each language:

```bash
# Parallel processing across languages (recommended)
python metaclip/metadata/build_ngram.py submitit

# Sequential processing (for testing)
python metaclip/metadata/build_ngram.py
```

**Special Tokenization**: Languages like Chinese (zh), Japanese (ja), Thai (th), Khmer (km), Lao (lo), Myanmar (my), and Tibetan (bo) use specialized tokenizers located in `metaclip/metadata/lang_tokenizers/`.

> Note: We use [`ckip-transformers`](https://github.com/ckiplab/ckip-transformers), a BERT-based word segmentation tool for Chinese. Running `build_ngram.py` for languages with Chinese characters (zh, zh_yue, zh_classical) will require GPUs, while other languages can run on CPUs.

### Step 4: Build Wikipedia Article Titles

Extract and rank Wikipedia article titles by pageview frequency:

```bash
# Parallel processing for multiple date ranges
python metaclip/metadata/build_title.py submitit

# Merge processed titles from multiple dates
python metaclip/metadata/build_title.py
```

### Step 5: Merge Metadata Sources

Combine all metadata sources (WordNet + unigrams + bigrams + titles) for each language:

```bash
python metaclip/metadata/build_metadata.py
```

**Per-Language Limits**:
- **WordNet synsets**: No limit (language-dependent)
- **Unigrams**: sorted by frequency, take top 10% of available unigrams per language, max at 251,465 terms (number of English unigrams)
- **Bigrams**: sorted by PMI, take top "x" bigrams per language, where x="40% of number of unigrams in the same language", max at 100,646 terms (number of English bigrams)
- **Titles**: sorted by pageviews, take the top 76% of available titles per language, max at 61,235 titles (number of English titles)

**Special Handling**:
- Non-space languages (Chinese, Japanese, etc.): Bigrams are concatenated without spaces
- Punctuation-only terms are filtered out
- Terms >256 characters are excluded

## Phase 2: Language ID Integration and Automaton Building

### Step 6: Align Language ID to Wikipedia Metadata

Create one-to-one mapping between Language ID detected languages and Wikipedia metadata:

```bash
python metaclip/metadata/align_lid_to_wiki.py
```

**Process**:
1. Maps LID language codes to Wikipedia language codes using `metaclip/curation/substr_matching.py:lid_to_wiki`
2. Merges related language variants (e.g., zh_classical, zh_yue → zh)
3. Creates unified metadata sets for each LID language
4. Builds Aho-Corasick automatons for fast substring matching

**Output**:
- `data/metadata/{lang_code}.json`: Merged metadata lists
- `data/metadata/{lang_code}.pkl`: Compiled automatons for fast matching

## Phase 3: Data Curation Pipeline

### Step 7: Substring Matching (Stage 1)

Count metadata matches in your dataset:

```bash
python metaclip/curation/curate.py count_per_shard
```

**Process**:
1. Reads text data from `data/pkg_json/{shard_id}.json`
2. Filters valid samples using `data/valid_uuids/{shard_id}.json`
3. Detects language using Language ID
4. Maps detected language to metadata using `lid_langcode_to_metadata_langcode()`
5. Performs substring matching using appropriate automaton
6. Counts matches per metadata entry per language

**Output**: `data/index_json/{shard_group}.npz` containing match counts

### Step 8: Global Count Aggregation (Stage 2)

Compute global statistics and entry probabilities:

```bash
python metaclip/curation/curate.py
```

**Process**:
1. Aggregates counts across all data shards
2. Computes tail probability `p` for English (reference: MetaCLIP t=170,000 vs original CLIP t=20,000)
3. Applies same tail probability to all other languages using `p_to_t()`
4. Converts counts to sampling probabilities using `count_to_prob()`

**Key Formula**: For entries with count < t: prob = 1.0; otherwise prob = t/count

**Output**: `data/index_json/per_lang_prob/{t}_{lang_id}.npy` containing entry probabilities

### Step 9: Balanced Sampling (Stage 3)

Curate image-text pairs using computed probabilities:

```bash
python metaclip/curation/curate.py curate
```

**Process**:
1. For each image with multiple text candidates:
   - Randomly select one text description
   - Get matched metadata entries for detected language
   - Compute curation probability: `prob = 1 - ∏(1 - entry_prob_i)`
   - Sample based on computed probability
2. Save curated pairs to `data/curated_index_json/`
