# Copyright (c) Meta Platforms, Inc. and affiliates


print("Loading ja tokenizer...", end=' ')
import MeCab
import ipadic
from typing import List

tagger = MeCab.Tagger(ipadic.MECAB_ARGS + " -Owakati")


print("ja tokenizer ready!")
def tokenize(
        texts: List[str]
) -> List[List[str]]:
    return [
        tagger.parse(text).split() for text in texts
    ]