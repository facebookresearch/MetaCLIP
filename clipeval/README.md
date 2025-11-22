# CLIPEval

We aggregate CLIP related benchmarks under `clipeval`. After configuration of thirdparty code, simply run `clipeval/eval_all.py` will launch evaluation on all models.


## SLIP

We refactored Meta CLIP 1's SLIP eval into `clipeval/slip`.

## XM3600

Download `data/XM3600/images` and `data/XM3600/captions.jsonl` from [this page](https://google.github.io/crossmodal-3600) and config your `data_dir` in `clipeval/xm3600/eval_xm3600.py`.

```bash
mkdir -p external
cd external
git clone https://github.com/google-research/big_vision.git
cd -  # root of this repo
```

## CVQA

```bash
pip install datasets  # we use 3.5.0
```
