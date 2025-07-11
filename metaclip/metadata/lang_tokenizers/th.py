# Copyright (c) Meta Platforms, Inc. and affiliates


print('Loading th tokenizer...', end=' ')

from thai_segmenter import tokenize as thai_segmenter_tokenize
from typing import List

print('th tokenizer ready!')
def tokenize(
        texts: List[str],
) -> List[List[int]]:
    return [thai_segmenter_tokenize(text) for text in texts]
