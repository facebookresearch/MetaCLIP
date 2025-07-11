# Copyright (c) Meta Platforms, Inc. and affiliates


print('Loading zh tokenizer...', end=' ')

from ckip_transformers.nlp import CkipWordSegmenter
from typing import List

ws_driver  = CkipWordSegmenter(model="bert-base", device=0)


print('zh tokenizer ready!')
def tokenize(
    texts: List[str]
) -> List[List[str]]:
    return ws_driver(texts)
