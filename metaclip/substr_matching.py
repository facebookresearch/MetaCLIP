# Copyright (c) Meta Platforms, Inc. and affiliates


spaced_metadata = None

def spacing(text):
    puncts_to_wrap = [",", ".", ";", ":", "?", "!", "`"]
    chars_to_space = ["\t", "\n", "\r"]

    spaced_text = f" {text} "
    for punct_to_wrap in puncts_to_wrap:
        spaced_text = spaced_text.replace(
            punct_to_wrap, f" {punct_to_wrap} "
        )
    for char_to_space in chars_to_space:
        spaced_text = spaced_text.replace(char_to_space, " ")
    return spaced_text


def substr_matching(text, metadata):
    global spaced_metadata
    if spaced_metadata is None:
        spaced_metadata = []
        for entry in metadata:
            spaced_metadata.append(f" {entry} ")
    text = spacing(text)
    matched_entry_ids = []
    for entry_id, entry in enumerate(spaced_metadata):
        if entry in text:
            matched_entry_ids.append(entry_id)
    return matched_entry_ids
