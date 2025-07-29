import json
import string

from datetime import datetime


def wordnet_synsets():
    from nltk.corpus import wordnet as wn
    entries = []
    for ss in wn.all_synsets():
        name = ss.name()
        dot_idx = name.find(".")
        name = name[:dot_idx].replace("_", " ")
        entries.append(name)
    return entries


def wiki_unigram(thres=100):
    entries = []
    with open("data/wiki/enwiki-unigram.txt") as fr:
        for line in fr:
            name, count = line.strip().split()
            count = int(count)
            if count >= thres:  # at least
                entries.append(name)
    return entries


def wiki_bigrams():
    import os
    import gzip

    if not os.path.exists("data/wiki/bigram_pmi_cache.txt.gz"):
        from nltk.probability import FreqDist

        word_fd = FreqDist()

        with gzip.open("data/wiki/1gram.txt.gz") as fr:
            for line in fr:
                count, name = line.decode().split("\t")
                count = int(count)
                name = name.strip()
                if len(name) > 0 and count > 0:
                    word_fd[name] = count

        bigram_fd = FreqDist()
        missing_word_count, total_count = 0, 0
        with gzip.open("data/wiki/2gram.txt.gz") as fr:
            for line in fr:
                count, word1, word2 = line.decode().split("\t")
                count = int(count)
                word1 = word1.strip()
                word2 = word2.strip()
                if len(word1) > 0 and len(word2) > 0 and count > 0:
                    total_count += 1
                    if word1 not in word_fd or word2 not in word_fd:
                        missing_word_count += 1
                        if missing_word_count % 500000 == 0:
                            print("missing words in unigram", line.decode())
                        continue
                    bigram_fd[(word1, word2)] = count

                    if len(bigram_fd) % 500000 == 0:
                        print("sample bi-gram", word1, word2)

        print(f"bigram stats: {missing_word_count} / {total_count}")

        from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder(word_fd, bigram_fd)
        with gzip.open("data/wiki/bigram_pmi_cache.txt.gz", "wb") as fw:
            for query in finder.score_ngrams(bigram_measures.pmi):
                fw.write(f"{query[1]}\t{query[0][0]}\t{query[0][1]}\n".encode())


    pmi_thres = 30.
    print(f"use pmi_thres={pmi_thres}")
    entries = []
    with gzip.open("data/wiki/bigram_pmi_cache.txt.gz") as fr:
        for line in fr:
            pmi, word1, word2 = line.decode().strip().split("\t")
            pmi = float(pmi)
            if pmi >= pmi_thres:
                entries.append(f"{word1} {word2}")
    return entries


def wiki_title(budget=100000):
    import urllib.request
    import os
    import gzip

    from collections import defaultdict

    if not os.path.exists("data/wiki/title_counts.json"):
        dates = [
            "20180419",
            "20180510",
            "20180530",
            "20180914",
            "20181119",
            "20190406",
            "20190928",
            "20191026",
            "20191208",
            "20200417",
            "20200610",
            "20200813",
            "20200824",
            "20201018",
            "20210112",
            "20210123",
            "20210305",
            "20210920",
            "20211127",
            "20211211",
            "20220116",
            "20220322",
            "20220430",
            "20220529",
            "20220618",
            "20220829"
        ]

        title_counts = defaultdict(int)

        for date in dates:
            print("date", date)
            for idx in range(0, 240000, 10000):
                fn = f"pageviews-{date}-{idx:06}.gz"
                local_path = f"data/wiki/pageviews/{fn}"
                if not os.path.exists(local_path):
                    urllib.request.urlretrieve(f"https://dumps.wikimedia.org/other/pageviews/{date[:4]}/{date[:4]}-{date[4:6]}/{fn}", local_path)
                with gzip.open(local_path) as fr:
                    for idx, line in enumerate(fr):
                        line = line.decode().strip()
                        if line.startswith("en "):
                            orgin, title, count, hour = line.split(" ")
                            assert orgin == "en", orgin
                            title = title.strip().replace("_", " ")
                            count = int(count)
                            count_filter = int(count / 50)
                            if count_filter > 0 and ":" not in title:
                                if title not in title_counts and (len(title_counts)+1) % 10000 == 0:
                                    print("len(title_counts)", len(title_counts))
                                title_counts[title] += count

        with open("data/wiki/title_counts.json", "w") as fw:
            json.dump(title_counts, fw)


    with open("data/wiki/title_counts.json") as fr:
        title_counts = json.load(fr)

    view_thres = 70
    print(f"use view_thres={view_thres}")

    entries = []
    for title, count in title_counts.items():
        if count >= view_thres:
            entries.append(title)    
    return entries


def main():
    num_entries = 500000

    forbidden = set(string.punctuation)
    sources = {"wordnet": wordnet_synsets, "wiki_unigram": wiki_unigram, "wiki_bigrams": wiki_bigrams, "wiki_title": wiki_title}

    entries = set([str(ix) for ix in range(100)])

    for source_name in sources:
        source_entries = set(sources[source_name]())
        for entry in source_entries:
            if len(entry) > 0 and entry not in forbidden:
                entries.add(entry)

            if len(entries) >= num_entries:
                today = datetime.today().strftime('%Y-%m-%d')
                with open(f"metadata_{today}.json", "w") as fw:
                    json.dump(list(entries), fw)
        print(f"after adding {source_name}: len(entries)={len(entries)}.")


if __name__ == "__main__":
    main()
