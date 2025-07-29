# Copyright (c) Meta Platforms, Inc. and affiliates

try:
    import ahocorasick
except ImportError:
    print("cannot import ahocorasick, try `pip install pyahocorasick`")


from substr_matching import spacing


automaton = None
spaced_metadata = None

def initialize_automaton(metadata):
    automaton = ahocorasick.Automaton()
    for idx, key in enumerate(spaced_metadata):
        automaton.add_word(key, (idx, key))
    automaton.make_automaton()
    return automaton


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
