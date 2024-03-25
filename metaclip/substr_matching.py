# Copyright (c) Meta Platforms, Inc. and affiliates
import ahocorasick


automaton = None
spaced_metadata = None

def initialize_automaton(metadata):
    automaton = ahocorasick.Automaton()
    for idx, key in enumerate(spaced_metadata):
        automaton.add_word(key, (idx, key))
    automaton.make_automaton()
    return automaton

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
    global spaced_metadata, automaton
    if spaced_metadata is None:
        spaced_metadata = [f" {entry} " for entry in metadata]
    text = spacing(text)
    if automaton is None:
        automaton = initialize_automaton(metadata)
    matched_entry_ids = set()
    for end_index, (entry_id, original_value) in automaton.iter(text):
        matched_entry_ids.add(entry_id)
    return list(matched_entry_ids)

