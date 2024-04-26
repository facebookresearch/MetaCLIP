# Demo Setup

## Install miniconda on Mac

[Miniconda Installation Guide](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)

For example on a M1 chip:

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
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    -c anaconda
conda activate metaclip
```

install gradio

```bash
pip install gradio
```

## Run Gradio

```bash
git clone https://github.com/facebookresearch/MetaCLIP.git
cd MetaCLIP
git checkout gradio
git pull
```


```bash
python app.py
```

follow the instruction on the terminal and open a browser.
