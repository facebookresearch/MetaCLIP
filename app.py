import json
import numpy as np
import gradio as gr

from metaclip.substr_matching import substr_matching
from metaclip.balancing import balance_sampling


entry_count = None
metadata = None

def init_demo():
    global metadata
    with open("metadata.json") as f:
        metadata = json.load(f)
    
    # entry counts for our 1.6B(pool) -> 400M(curated); please check balance_sampling:main and substr match and count on your own data.
    with open("metaclip/entry_counts_400m.json") as f:
        entry_count_json = json.load(f)
    global entry_count
    entry_count = np.array([entry_count_json[entry] for entry in metadata], dtype=np.uint64)  # uint64 to be safe for scaling.


def curation(text):
    t = 20000  # TODO: make this part of the UI
    entry_count[entry_count < t] = t
    entry_prob = t / entry_count

    matched_entry_ids = substr_matching(text, metadata)
    curation_prob = min(entry_prob[matched_entry_ids].sum(), 1.0)
    curated = balance_sampling(matched_entry_ids, entry_prob)
    
    return f"curation_prob={curation_prob:.3f}, curated={curated}"


init_demo()

demo = gr.Interface(fn=curation, inputs="text", outputs="text")
    
if __name__ == "__main__":
    demo.launch(show_api=False)  
