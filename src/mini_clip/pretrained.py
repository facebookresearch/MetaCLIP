# Copyright (c) Meta Platforms, Inc. and affiliates

import hashlib
import os
import urllib
import warnings

from tqdm import tqdm

_RN50 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    yfcc15m="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt",
    cc12m="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt"
)

_RN50_quickgelu = dict(
    openai="https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    yfcc15m="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt",
    cc12m="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt"
)

_RN101 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    yfcc15m="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt"
)

_RN101_quickgelu = dict(
    openai="https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    yfcc15m="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt"
)

_RN50x4 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
)

_RN50x16 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
)

_RN50x64 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
)

_VITS16_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_s16_224px_worldwide.pt", "951f60fcba8aa57ddc993ba995653d0a3320c3b129a6824ab1a76d3a3eb1a438"),
)

_VITS16_384_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_s16_384px_worldwide.pt", "834f3ebc39c2356136387c697c50c864ba4da24bf4fb453068e2b900dddc5d91"),
)

_VITS16_mT5_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_s16_224px_mt5_worldwide.pt", "c0c269758b98fe754be491f3cc70d753e0d1f0cf070fe4e9d042da4378feefd9"),
)

_VITM16_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_m16_224px_worldwide.pt", "2112c62563f8de00544a7578350c6255dcad87f6bcd3305b13c7ecb30a104dbf"),
)

_VITM16_384_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_m16_384px_worldwide.pt", "6699bc5265dd14abdb2e61cb4d089205c4b927e7bf8b77d493c9faf18d9236c2"),
)

_VITM16_mT5_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_m16_224px_mt5_worldwide.pt", "976d6cbfe35529528abede17f6b7a2e8120b5fd5f9ada6e362e2f80e919f556f"),
)

_VITB32 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    laion2b_e16="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth",
    laion400m_e31="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
    laion400m_e32="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
)

_VITB32_quickgelu = dict(
    openai="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    laion400m_e31="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
    laion400m_e32="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
    metaclip_400m=("https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_400m.pt", "3c68642594a329afc1ec0fe489ee2b58ab19c9d0556ccf7c404a59baa0762d71"),
    metaclip_2_5b=("https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_fullcc2.5b.pt", "885b7ec11fe07a9826e2e6812d70e5011918e32fe9b12136b49d5dded92b4386"),
    metaclip_fullcc=("https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_fullcc2.5b.pt", "885b7ec11fe07a9826e2e6812d70e5011918e32fe9b12136b49d5dded92b4386"),
    metaclip400m=("https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_400m.pt", "3c68642594a329afc1ec0fe489ee2b58ab19c9d0556ccf7c404a59baa0762d71"),
    metaclip2_5b=("https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_fullcc2.5b.pt", "885b7ec11fe07a9826e2e6812d70e5011918e32fe9b12136b49d5dded92b4386"),
)

_VITB32_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_b32_224px_worldwide.pt", "d8e10eb67b4300b3509c63d56337efd68cf385e9fd872da889c359c045fd629c"),
)

_VITB32_384_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_b32_378px_worldwide.pt", "5056f3e0cd4409d01d44887753ed296e4d1b3a358547ce6435e1d1886b8ba5ea"),
)

_VITB32_mT5_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_b32_224px_mt5_worldwide.pt", "8795b43872bcc5babe2feabd90bfbb53bd4470a39bd44c79b84f9be758fc3a1d"),
)

_VITB16 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    laion400m_e31="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e31-00efa78f.pt",
    laion400m_e32="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e32-55e67d44.pt",
)

_VITB16_quickgelu = dict(
    metaclip_400m=("https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt", "68dfb5996c52a8f4fecb9bd16601e97e1895236645082778bd9cede8429a8d49"),
    metaclip_2_5b=("https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_fullcc2.5b.pt", "512ea0fb9f2cf88d027e96e4674247a1a91a96af18abc2e2fcdb8008c551e04b"),
    metaclip_fullcc=("https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_fullcc2.5b.pt", "512ea0fb9f2cf88d027e96e4674247a1a91a96af18abc2e2fcdb8008c551e04b"),
    metaclip400m=("https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt", "68dfb5996c52a8f4fecb9bd16601e97e1895236645082778bd9cede8429a8d49"),
    metaclip2_5b=("https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_fullcc2.5b.pt", "512ea0fb9f2cf88d027e96e4674247a1a91a96af18abc2e2fcdb8008c551e04b"),
)

_VITB16_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_b16_224px_worldwide.pt", "61c68cbd496b92669297bec5d37e5a4a9efbd05df79e9224b21dc43666b8dc8a"),
)

_VITB16_384_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_b16_384px_worldwide.pt", "f9a1f207ac79f78fe99dd252ded562031b587595b19a8b8ca36b89d36c6eb347"),
)

_VITB16_PLUS_240 = dict(
    laion400m_e31="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt",
    laion400m_e32="https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e32-699c4b84.pt",
)

_VITL14 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    laion400m_e31='https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e31-69988bb6.pt',
    laion400m_e32='https://github.com/mlfoundations/mini_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e32-3d133497.pt',
)

_VITL14_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_l14_224px_worldwide.pt", "3ffe7d5099e088b5c27061fb1908b70326fb1647aa4dd804da02b4caf8894882"),
)

_VITL14_quickgelu = dict(
    metaclip_400m=("https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_400m.pt", "51c782959f920b030779e494517b8d545f56794df6b0a2796a4c310455a361be"),
    metaclip_2_5b=("https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt", "ce24750710544ee288ef0abdead2016730da1893a1d07447bda3a75e1c148f97"),
    metaclip_fullcc=("https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt", "ce24750710544ee288ef0abdead2016730da1893a1d07447bda3a75e1c148f97"),
    metaclip400m=("https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_400m.pt", "51c782959f920b030779e494517b8d545f56794df6b0a2796a4c310455a361be"),
    metaclip2_5b=("https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt", "ce24750710544ee288ef0abdead2016730da1893a1d07447bda3a75e1c148f97"),
)

_VITL14_336 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
)

_VITH14_quickgelu = dict(
    metaclip2_5b=("https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_fullcc2.5b.pt", "1286807d5cc8d9a0b12563b47474efb53b9522eb3d7eac5a9a5d39c3a776ad5c"),
    metaclip_2_5b=("https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_fullcc2.5b.pt", "1286807d5cc8d9a0b12563b47474efb53b9522eb3d7eac5a9a5d39c3a776ad5c"),
    metaclip_fullcc=("https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_fullcc2.5b.pt", "1286807d5cc8d9a0b12563b47474efb53b9522eb3d7eac5a9a5d39c3a776ad5c"),
)

_VITH14_quickgelu_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_h14_quickgelu_224px_worldwide.pt","9d28636c949d20d9c2e4cb4fd3824082e12d383851b6bfcd65eb33e165c63e37"),
)

_VITH14 = dict(
    metaclip_v1_2_altogether=("https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_v1.2_altogether.pt", "c4ee0a62c58a38867df142d7be3b0aa1988d6ac49293971b177a03ead6092cc6"),
)

_VITH14_378_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_h14_378px_worldwide.pt", "914d25384ad6655af5309e875abc17619f4a4ce9c2030675bcec7626214341a9"),
)

_VITbigG14_quickgelu = dict(
    metaclip2_5b=("https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt", "5fe2b83c7439e0caa2c855dec9a2eaa54f17f3ced288218564b640ca7953447f"),
    metaclip_2_5b=("https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt", "5fe2b83c7439e0caa2c855dec9a2eaa54f17f3ced288218564b640ca7953447f"),
    metaclip_fullcc=("https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt", "5fe2b83c7439e0caa2c855dec9a2eaa54f17f3ced288218564b640ca7953447f"),
)

_VITbigG14_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_bigG14_224px_worldwide.pt", "aa87c36904aa607b0f58a6537d965ce08e62073e4204acc38e8a5c586cc83624"),
)

_VITbigG14_378_worldwide = dict(
    metaclip2_worldwide=("https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_bigG14_378px_worldwide.pt", "42166f47ec5b52771865931c266d5f09b67e031cea2daa6df68bee3c7ca0baa6"),
)


_PRETRAINED = {
    "RN50": _RN50,
    "RN50-quickgelu": _RN50_quickgelu,
    "RN101": _RN101,
    "RN101-quickgelu": _RN101_quickgelu,
    "RN50x4": _RN50x4,
    "RN50x16": _RN50x16,
    "RN50x64": _RN50x64,
    "ViT-S-16-worldwide": _VITS16_worldwide,
    "ViT-S-16-384-worldwide": _VITS16_384_worldwide,
    "ViT-S-16-mT5-worldwide": _VITS16_mT5_worldwide,    
    "ViT-M-16-worldwide": _VITM16_worldwide,    
    "ViT-M-16-384-worldwide": _VITM16_384_worldwide,
    "ViT-M-16-mT5-worldwide": _VITM16_mT5_worldwide,
    "ViT-B-32": _VITB32,
    "ViT-B-32-quickgelu": _VITB32_quickgelu,
    "ViT-B-32-worldwide": _VITB32_worldwide,
    "ViT-B-32-384-worldwide": _VITB32_384_worldwide,
    "ViT-B-32-mT5-worldwide": _VITB32_mT5_worldwide,
    "ViT-B-16": _VITB16,
    "ViT-B-16-quickgelu": _VITB16_quickgelu,
    "ViT-B-16-worldwide": _VITB16_worldwide,
    "ViT-B-16-384-worldwide": _VITB16_384_worldwide,
    "ViT-B-16-plus-240": _VITB16_PLUS_240,
    "ViT-L-14": _VITL14,
    "ViT-L-14-quickgelu": _VITL14_quickgelu,
    "ViT-L-14-worldwide": _VITL14_worldwide,
    "ViT-L-14-336": _VITL14_336,
    "ViT-H-14-quickgelu": _VITH14_quickgelu,
    "ViT-H-14": _VITH14,
    "ViT-H-14-quickgelu-worldwide": _VITH14_quickgelu_worldwide,
    "ViT-H-14-378-worldwide": _VITH14_378_worldwide,
    "ViT-bigG-14-quickgelu": _VITbigG14_quickgelu,
    "ViT-bigG-14-worldwide": _VITbigG14_worldwide,
    "ViT-bigG-14-378-worldwide": _VITbigG14_378_worldwide,
}


def list_pretrained(as_str: bool = False):
    """ returns list of pretrained models
    Returns a tuple (model_name, pretrain_tag) by default or 'name:tag' if as_str == True
    """
    return [':'.join([k, t]) if as_str else (k, t) for k in _PRETRAINED.keys() for t in _PRETRAINED[k].keys()]


def list_pretrained_tag_models(tag: str):
    """ return all models having the specified pretrain tag """
    models = []
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)
    return models


def list_pretrained_model_tags(model: str):
    """ return all pretrain tags for the specified model architecture """
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags


def get_pretrained_url(model: str, tag: str):
    if model not in _PRETRAINED:
        return ''
    model_pretrained = _PRETRAINED[model]
    tag = tag.lower()
    if tag not in model_pretrained:
        return ''
    return model_pretrained[tag]


def download_pretrained(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    elif isinstance(url, tuple):
        assert len(url) == 2, "url w/ sha256 hash must be in form (url, sha256) tuple."
        expected_sha256 = url[1]
        url = url[0]
    else:
        expected_sha256 = ''

    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target
