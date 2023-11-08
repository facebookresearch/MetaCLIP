# Copyright (c) Meta Platforms, Inc. and affiliates

import math
import json
import time
import os
import logging
import gzip
import hashlib

from urllib.parse import urljoin
from collections import defaultdict
from multiprocessing import Process, Manager, Pool
from substr_matching import substr_matching


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def gen_uuid(url):
    return hashlib.sha224(bytes(url, encoding="utf-8")).hexdigest()[:24]


def lid(text):
    import fasttext
    fasttext.FastText.eprint = lambda x: None
    from ftlangdetect import detect
    return detect(text=text, low_memory=True)["lang"]


def process_data(raw_data, metadata):
    matched_data = []
    for pair in raw_data:
        texts = []
        for key_text in pair["texts"]:
            text_key, text = key_text
            if len(text) == 0:
                continue
            matched_entry_ids = substr_matching(text, metadata)
            if len(matched_entry_ids) > 0:
                orig_text = key_text[1]
                texts.append([text_key, orig_text, matched_entry_ids])

        if len(texts) > 0:
            pair["texts"] = texts
            matched_data.append(pair)
    return matched_data


class CCCurator(object):
    KOI = ["alt", "title", "data-image-title"]

    def parse_html(self, html, target_uri, data):
        """open your browser, right-click and inspect"""
        raise NotImplementedError("derive this class a for specific type of parser.")

    def parse_htmls(self, html, target_uri, data):
        """open your browser, right-click and inspect"""
        raise NotImplementedError("derive this class a for specific type of parser.")

    @classmethod
    def save_json(cls, fn, data):
        from pathlib import Path
        Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
        with open(fn, "w") as fw:
            json.dump(data, fw, indent=1)

    @classmethod
    def normalize_url(cls, url, target_uri, strip_param=False):
        try:
            url = urljoin(target_uri, url)
            if strip_param:
                url = url.split("?")[0]
            return url
        except Exception:  # eg ed2k://
            return None

    def substrmatch(self, data, metadata, num_proc=20):
        pairs_per_proc = math.ceil(len(data) / num_proc)

        data_lists = [
            data[start : start + pairs_per_proc]
            for start in range(0, len(data), pairs_per_proc)
        ]

        entries_lists = [metadata] * len(data_lists)
        num_proc = len(data_lists)

        with Pool(num_proc) as p:
            results = p.starmap(process_data, zip(data_lists, entries_lists))

        data = []
        for result in results:
            data.extend(result)
        return data


class WARCCurator(CCCurator):
    def __init__(self, dedup=True, lid=True):
        self.dedup = dedup
        self.url_dedup = defaultdict(set)
        self.lid = lid

    def clean_dedup_cache(self):
        self.url_dedup = defaultdict(set)

    def parse(self, cc_file, verbose=False):
        from warcio.archiveiterator import ArchiveIterator
        data = []
        htmls = []
        start = time.time()

        with open(cc_file, 'rb') as stream:
            record_iter = ArchiveIterator(stream)
            if verbose:
                from tqdm import tqdm
                record_iter = tqdm(record_iter)
            for ix, record in enumerate(record_iter):
                if record.rec_type == 'response':
                    htmls.append((record.raw_stream.read().strip(), record.rec_headers.get_header('WARC-Target-URI')))
                    if len(htmls) >= 500:
                        # timeout a bad HTML page.
                        ret_data = Manager().list()
                        p = Process(target=self.parse_htmls, args=[htmls, ret_data])
                        p.start()
                        p.join(60*5)
                        if p.is_alive():
                            p.terminate()
                            print(f"timeout on {cc_file}")
                        htmls = []
                        data.extend(ret_data)

            if len(htmls) > 0:
                # timeout a bad HTML page.
                ret_data = Manager().list()
                p = Process(target=self.parse_htmls, args=[htmls, ret_data])
                p.start()
                p.join(60*5)
                if p.is_alive():
                    p.terminate()
                    logging.info(f"timeout on {cc_file}")
                htmls = []
                data.extend(ret_data)
        logging.info(f"{cc_file}: {time.time() - start} seconds, len(data)={len(data)}")
        return data

    def parse_htmls(self, htmls, data):
        for html, target_uri in htmls:
            self.parse_html(html, target_uri, data)

    def parse_html(self, html, target_uri, data):
        from selectolax.parser import HTMLParser
        tree = HTMLParser(html)
        for img_node in tree.tags("img"):
            texts = []
            url = None
            for key in img_node.attrs.keys():
                if key not in CCCurator.KOI + ["src"]:
                    continue

                text = img_node.attrs[key]
                if text is None:
                    continue
                text = text.strip().replace("\n", "").replace("\r", "")
                if len(text) == 0:
                    continue

                if key == "src":
                    url = CCCurator.normalize_url(text, target_uri)

                if key in CCCurator.KOI and (not self.lid or (self.lid and lid(text) == "en")):
                    texts.append([key, text])

            if url is None:
                continue
            rec = {"uuid": gen_uuid(url), "url": url, "texts": []}
            # dedup on URL.
            if self.dedup:
                text_sets = self.url_dedup[rec["uuid"]]
                for text in texts:
                    if text[1] not in text_sets:
                        text_sets.add(text[1])
                        rec["texts"].append(text)
            else:
                rec["texts"] = texts
            if len(rec["texts"]) > 0:
                data.append(rec)


class WATCurator(CCCurator):
    def __init__(self, dedup=True, lid=True):
        self.dedup = dedup
        self.url_dedup = defaultdict(set)
        self.lid = lid

    def parse(self, cc_file, verbose=False):
        data = []
        num_records = 0
        with gzip.open(cc_file) as fr:
            looking_for_json = False
            for line in fr:
                try:
                    line = line.decode().strip()
                except Exception as e:
                    print(e)
                    continue
                if line.startswith("WARC-Target-URI"):
                    target_uri = line[len("WARC-Target-URI: ") :]
                    looking_for_json = True
                if looking_for_json and line.startswith("{"):
                    looking_for_json = False
                    self.parse_json(line, target_uri, data)
                    num_records += 1
        return data

    def parse_json(self, line, target_uri, data):
        try:
            record_data = json.loads(line)
        except Exception as e:  # pylint: disable=bare-except
            print(f"one record failed: {e}")
            return
        envelope = record_data["Envelope"]
        payload = envelope["Payload-Metadata"]
        if "HTTP-Response-Metadata" not in payload:
            return
        http_resp = payload["HTTP-Response-Metadata"]
        if "HTML-Metadata" not in http_resp:
            return
        metadata = http_resp["HTML-Metadata"]
        if "Links" not in metadata:
            return
        data.extend(self.extract_images_from_links(metadata["Links"], target_uri))

    def extract_images_from_links(
        self, links, target_uri
    ):
        results = []
        for link in links:
            if link is None or "path" not in link or link["path"] is None:
                continue
            if not link["path"].startswith("IMG@/src") or "url" not in link:
                continue
            url = CCCurator.normalize_url(link["url"], target_uri)
            if url is None:
                continue
            uuid = gen_uuid(url)
            if uuid is None:
                continue

            texts = []
            for key in CCCurator.KOI:
                if key not in link:
                    continue
                text = link[key]
                if text is None:
                    continue
                text = text.replace("\n", " ").replace("\r", " ").strip()
                if len(text) == 0:
                    continue
                texts.append([key, text])
            
            rec = {"uuid": uuid, "url": url, "texts": []}
            # dedup on URL.
            if self.dedup:
                text_sets = self.url_dedup[rec["uuid"]]
                for text in texts:
                    if text[1] not in text_sets:
                        text_sets.add(text[1])
                        rec["texts"].append(text)
            else:
                rec["texts"] = texts
            if len(rec["texts"]) > 0:
                results.append(rec)

        return results


def process(cc_file, output_file):
    if cc_file.endswith("wat.gz"):
        parser = WATCurator(dedup=True, lid=True)
    elif cc_file.endswith("warc.gz"):
        parser = WARCCurator(dedup=True, lid=True)
    else:
        raise ValueError(f"unknown cc extension {cc_file}")

    data = parser.parse(cc_file)

    with open("metadata.json") as f:
        metadata = json.load(f)
    data = parser.substrmatch(data, metadata)
    parser.save_json(output_file, data)


if __name__ == '__main__':
    import sys
    cc_file = sys.argv[1]
    output_file = sys.argv[2]
    process(cc_file, output_file)
