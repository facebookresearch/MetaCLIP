# MetaCLIP Data Pipeline

This is a minimal demo/skeleton code of CLIP curation, please check Algorithm 1 in [MetaCLIP paper](https://arxiv.org/pdf/2309.16671.pdf).
**This is not the pipeline used to collect data in paper**.

## Part 0 Build Metadata (optional)

see [README](README_metadata.md).

## Part 1 Sub-string matching

The key function of sub-string matching is in [substr_matching](substr_matching.py).
We also include a CommonCrawl WARC parser (in `cc_matching.py`) that requires the following package for fast HTML parsing. 

```bash
pip install warcio
pip install selectolax
pip install fasttext-langdetect  # for LID
pip install tqdm
```

For customized HTML parser, check [selectolax doc](https://selectolax.readthedocs.io/en/latest/parser.html) for more details.

### Usage

The parser supports both WAT and WARC formats (WAT is pre-parsed format from WARC, we expect 1% loss of image-text pairs).
[Get a CommonCrawl WAT/WARC file](https://commoncrawl.org/get-started), S3 is recommended, here is a quick http example:

```bash
mkdir -p data/CC/warc; mkdir -p data/CC/wat; mkdir -p data/CC/matched
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz -O data/CC/warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz
```

Then 

```bash
python metaclip/cc_matching.py data/CC/warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz data/CC/matched/CC-MAIN-20210723143921-20210723173921-00000.warc.gz.json
```

Want a distributed system to parse the full CC and download a dataset? consider to integrate `substr_matching.py` and `balancing.py` into a open source system: [cc2dataset](https://github.com/rom1504/cc2dataset/tree/main) and [img2dataset](https://github.com/rom1504/img2dataset).

## Part 2 Balancing (expected after image downloading/NSFW/dedup)


```bash
mkdir -p data/CC/balanced
python metaclip/balancing.py data/CC/matched data/CC/balanced 20000  # the magic 20k !
```

We expect balancing is the last step to ensure training data distribution. If you want such a run before image downloading/NSFW/dedup etc., please increase 20000 to a larger number and rerun balancing after getting images to accomendate loss of URL-text pairs. 


## Numpy Impl. 

We also provide a numpy impl. of the algorithm, which is close to the impl. in the paper.

```bash
python metaclip/pipeline.py metaclip_400m substr_indexing
python metaclip/pipeline.py metaclip_400m entry_count
python metaclip/pipeline.py metaclip_400m balance_sampling
```

## TODO: 
- integrate numpy impl. w/ WARC/WAT parser
