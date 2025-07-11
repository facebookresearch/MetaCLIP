# Copyright (c) Meta Platforms, Inc. and affiliates


import json
import string
import tqdm

from datetime import datetime


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


extra_dates = [
    "20220721",
    "20230123",
    "20230312",
    "20230516",
    "20230721",
    "20230913",
    "20231104",
    "20240130",
    "20240215",
    "20240401",
    "20240528",
]


def process_date(output_path, date):
    import urllib.request
    import os
    import gzip

    from pathlib import Path
    from collections import defaultdict

    if os.path.exists(f"{output_path}/title_counts/title_counts_{date}.json"):
        return

    title_per_lang = defaultdict(lambda : defaultdict(int))
    print("date", date)
    for idx in range(0, 240000, 10000):
        fn = f"pageviews-{date}-{idx:06}.gz"
        local_path = f"{output_path}/pageviews/{fn}"
        if not os.path.exists(local_path):
            Path(f"{output_path}/pageviews").mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://dumps.wikimedia.org/other/pageviews/{date[:4]}/{date[:4]}-{date[4:6]}/{fn}", local_path)
        print(local_path)
        with gzip.open(local_path) as fr:
            for idx, line in enumerate(fr):
                line = line.decode().strip()
                if line.startswith('"" '):
                    continue
                line = line.split(" ")
                if len(line) != 4:
                    continue
                lang_code, title, count, hour = line

                if '.m' in lang_code:
                    lang_code = lang_code.replace('.m', '')

                title = title.strip().replace("_", " ")
                count = int(count)
                if count == 0 or ":" in title:
                    continue
                title_per_lang[lang_code][title] += count

    Path(f"{output_path}/title_counts").mkdir(parents=True, exist_ok=True)
    with open(f"{output_path}/title_counts/title_counts_{date}.json", "w") as fw:
        json.dump(title_per_lang, fw)


def wiki_multilingual_title(output_path):
    from collections import defaultdict
    from tqdm import tqdm
    from pathlib import Path

    all_title_per_lang = defaultdict(lambda : defaultdict(int))
    for date in tqdm(dates + extra_dates):
        with open(f"{output_path}/title_counts/title_counts_{date}.json") as f:
            title_per_lang = json.load(f)
        for lang_code in title_per_lang:
            for title in title_per_lang[lang_code]:
                all_title_per_lang[lang_code.split('.')[0]][title] += title_per_lang[lang_code][title]
    
    Path(f"{output_path}/title_per_lang").mkdir(parents=True, exist_ok=True)
    for lang_code in all_title_per_lang:
        wiki_lang_code = lang_code.replace("-", "_")
        with open(f"{output_path}/title_per_lang/{wiki_lang_code}.json", "w") as fw:
            json.dump(all_title_per_lang[lang_code], fw, ensure_ascii=False)


if __name__ == "__main__":
    output_path = "data/metadata_source/wiki_title"

    all_dates = dates + extra_dates

    import sys

    if len(sys.argv) == 1:  # merge
        wiki_multilingual_title(output_path)
    else:
        import os
        import submitit
        import sys
        import math

        job_plans = [
            ("data", len(all_dates)),
        ]

        for partition, array_size in job_plans:
            params = dict(
                name=f"title",
                gpus_per_node=0,
                mem_gb=1,
                cpus_per_task=4,
                nodes=1,
                slurm_partition=partition,
                timeout_min=4320,
            )

            executor = submitit.AutoExecutor(
                folder="submitit/%j"
            )
            executor.update_parameters(**params)

            jobs = []
            with executor.batch():
                for offset in range(array_size):
                    job = executor.submit(
                        process_date,
                        output_path,
                        all_dates[offset]
                    )
                    jobs.append(job)

            if len(jobs) > 0:
                print(partition, len(jobs), jobs[0].job_id, jobs[-1].job_id)
