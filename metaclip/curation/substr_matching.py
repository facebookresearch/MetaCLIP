import string
import ahocorasick


additional_punctuation = {
    k: 0
    for k in [
        "，",
        "。",
        "、",
        "；",
        "：",
        "？",
        "！",
        "“",
        "”",
        "‘",
        "’",
        "（",
        "）",
        "【",
        "】",
        "《",
        "》",
        "〈",
        "〉",
        "「",
        "」",
        "『",
        "』",
        "～",
        "—",
    ]
}

lid_to_wiki = {

}


no_space_languages = {
    'bo',
    'dz',
    'ja',
    'km',
    'lo',
    'my',
    'ryu',
    'th',
    'zh',
}


def LID_langcode_to_metadata_langcode(lang_id):
    """
    convert lang. code from LID (language identification) into lang. code of metadata.
    
    Parameters:
    lang_id (str): language code from LID;
    
    Returns:
    lang_id (str): language code of metadata;
    """
    return 'other' if lang_id == '' or lid_to_wiki[lang_id] == "N/A" else lid_to_wiki[lang_id]


def is_punctuation(char):
    return char in string.punctuation or char in additional_punctuation


def is_cjk_or_similar(char):
    code_point = ord(char)
    # CJK ranges
    if (
        0x4E00 <= code_point <= 0x9FFF
        or 0x3400 <= code_point <= 0x4DBF
        or 0x20000 <= code_point <= 0x2A6DF
        or 0x2A700 <= code_point <= 0x2B73F
        or 0x2B740 <= code_point <= 0x2B81F
        or 0x2B820 <= code_point <= 0x2CEAF
        or 0x2CEB0 <= code_point <= 0x2EBEF
        or 0xF900 <= code_point <= 0xFAFF
        or 0x2E80 <= code_point <= 0x2EFF
        or 0x2F00 <= code_point <= 0x2FDF
        or 0x2FF0 <= code_point <= 0x2FFF
    ):
        return True
    # Thai, Lao, Burmese, Khmer, Tibetan ranges
    elif (
        0x0E00 <= code_point <= 0x0E7F  # Thai
        or 0x0E80 <= code_point <= 0x0EFF  # Lao
        or 0x1000 <= code_point <= 0x109F  # Burmese
        or 0x1780 <= code_point <= 0x17FF  # Khmer
        or 0x0F00 <= code_point <= 0x0FFF  # Tibetan
    ):
        return True
    # Punctuation
    elif is_punctuation(char):
        return True
    else:
        return False


def initialize_automaton(spaced_metadata):
    automaton = ahocorasick.Automaton()
    for idx, key in spaced_metadata:
        automaton.add_word(key, (idx, key))
    automaton.make_automaton()
    return automaton


def spacing(text):
    puncts_to_wrap = [",", ".", ";", ":", "?", "!", "`"]
    chars_to_space = ["\t", "\n", "\r"]

    spaced_text = f" {text} "
    for punct_to_wrap in puncts_to_wrap:
        spaced_text = spaced_text.replace(punct_to_wrap, f" {punct_to_wrap} ")
    for char_to_space in chars_to_space:
        spaced_text = spaced_text.replace(char_to_space, " ")
    return spaced_text


def get_spaced_metadata_ml(metadata):
    spaced_metadata = []
    for idx, entry in enumerate(metadata):
        spaced_entry = entry
        if not is_cjk_or_similar(spaced_entry[0]):
            spaced_entry = f" {spaced_entry}"
        if not is_cjk_or_similar(spaced_entry[-1]):
            spaced_entry = f"{spaced_entry} "
        spaced_metadata.append((idx, spaced_entry))
    return spaced_metadata


def substr_match(lang_id, txt, automaton_dir, automaton_ml, matching_fn="iter"):
    if lang_id not in automaton_ml:
        print("init automaton", lang_id)
        with open(f'{automaton_dir}/{lang_id}.pkl', 'rb') as f:
            automaton = pickle.load(f)
        automaton_ml[lang_id] = automaton

    spaced_txt = spacing(txt)
    matched_entry_ids = set()
    fn = getattr(automaton_ml[lang_id], matching_fn)
    for _, (entry_id, _) in fn(spaced_txt):
        matched_entry_ids.add(entry_id)
    matched_entry_ids_list = list(matched_entry_ids)
    return matched_entry_ids_list
