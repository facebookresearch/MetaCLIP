# Demo Setup

## Install miniconda

```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

initialize bash

```bash
~/miniconda3/bin/conda init bash
```

## Install Conda Dependency

```bash
conda create -n metaclip python=3.10 pytorch torchvision tqdm ftfy braceexpand regex pandas submitit=1.2.1 \
    -c pytorch-nightly \
    -c nvidia \
    -c conda-forge \
    -c anaconda
```

install gradio

```bash
pip install gradio
```

## Run Gradio

```bash
python app.py
```
