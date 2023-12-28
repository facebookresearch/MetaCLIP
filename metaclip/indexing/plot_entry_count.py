import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


f, ax = plt.subplots(figsize=(12, 8), dpi=200)
sns.despine(f)


def count_cumsum_bal(args):
    # adjust color
    g = sns.lineplot(x=[0], y=[0], linewidth=0)
    g = sns.lineplot(x=[0], y=[0], linewidth=0)
    g = sns.lineplot(x=[0], y=[0], linewidth=0)

    word_counts = np.load(f"{args.meta_index_path}/entry_counts.npy")
    counts = np.sort(word_counts)

    cumsum_counts = np.cumsum(counts)
    y = cumsum_counts.tolist()

    g = sns.lineplot(x=list(range(len(y))), y=y, linewidth=4)
    plt.text(len(y)*0.95, 1.05*y[-1], r"t=$\infty$", horizontalalignment='left', size=18, color="blue")

    pts = [20000, 90000, 170000]

    for t in pts:
        _counts = np.array(counts)
        _counts[_counts > t] = t
        _cumsum = np.cumsum(_counts)
        y = _cumsum.tolist()
        g = sns.lineplot(x=list(range(len(y))), y=y, linewidth=6)
        plt.text(len(y), 1.05*(y[-1]), f"t={t}", horizontalalignment='left', size=18, color="blue")

    ax = plt.gca()
    # Set alpha transparency for the line
    line = ax.lines[-2].set_alpha(0.7)
    line = ax.lines[-1].set_alpha(0.7)  # Get the first line from the plot

    g.set_xlabel(f"Metadata Entries Sorted by Counts", fontsize=16)
    g.set_ylabel("Cumulative Entry Counts", fontsize=16)
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )


plots = {
    "count_cumsum_bal": count_cumsum_bal,
}

# python scripts/huxu/balancing/plot_entry_counting.py llama2c8_v01
import sys
sys.path.append("./")

from btm.utils import load_config
config = load_config(sys.argv[1])

count_cumsum_bal(config)
plt.savefig(os.path.join("plots", f"{sys.argv[1]}.jpg"))
# plt.savefig(os.path.join("plots/queries", f"{sys.argv[1]}.pdf"))
