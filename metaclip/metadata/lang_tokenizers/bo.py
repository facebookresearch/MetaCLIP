# Copyright (c) Meta Platforms, Inc. and affiliates


print("Loading bo tokenizer...", end=' ')

from botok import WordTokenizer
from botok.config import Config
from pathlib import Path
from typing import List

def get_tokens(wt, text):
    tokens = wt.tokenize(text, split_affixes=False)
    return [x['text'] for x in tokens]

config = Config(dialect_name="general", base_path= Path.home())
wt = WordTokenizer(config=config)

print("bo tokenizer ready!")
def tokenize(
        texts: List[str]
) -> List[List[str]]:
    return [
        get_tokens(wt, text) for text in texts
    ]