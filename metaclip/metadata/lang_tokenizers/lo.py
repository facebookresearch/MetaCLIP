# Copyright (c) Meta Platforms, Inc. and affiliates


print('Loading lo tokenizer...', end=' ')

from laonlp.tokenize import word_tokenize
from typing import List

print('lo tokenizer ready!')
def tokenize(
        texts: List[str]
) -> List[List[str]]:
    return [
        word_tokenize(text) for text in texts
    ]