# Copyright (c) Meta Platforms, Inc. and affiliates


import hashlib
import json
import os

from urllib.parse import urljoin, urlparse


def lid(text):
    import fasttext
    fasttext.FastText.eprint = lambda x: None
    from ftlangdetect import detect
    return detect(text=text, low_memory=True)["lang"]


class WATParser(object):
    KOI = ["alt", "title", "data-image-title"]

    def parse_wat(self, wat_fn, snapshot, warc_id):
        data = []
        num_records = 0
        with gzip.open(wat_fn) as fr:
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
                    self.parse_json(line, target_uri, snapshot, warc_id, data)
                    num_records += 1
        return data

    def parse_json(self, line, target_uri, snapshot, warc_id, data):
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
        data.extend(
            self.extract_images_from_links(
                metadata["Links"], target_uri, snapshot, warc_id
            )
        )

    def extract_images_from_links(
        self, links, target_uri, snapshot, warc_id
    ):  # follows our WARCParser impl.
        results = []
        for link in links:
            if link is None or "path" not in link or link["path"] is None:
                continue
            if not link["path"].startswith("IMG@/src") or "url" not in link:
                continue
            url = WATParser.normalize_url(link["url"], target_uri)
            if url is None:
                continue
            uuid = WATParser.gen_uuid(url)
            if uuid is None:
                continue

            texts = []
            for key in WATParser.KOI:
                if key not in link:
                    continue
                text = link[key]
                if text is None:
                    continue
                text = text.replace("\n", " ").replace("\r", " ").strip()
                if len(text) == 0:
                    continue
                texts.append([key, text, lid(text)])
            if len(texts) > 0:
                results.append(
                    {
                        "url": url,
                        "texts": texts,
                        "uuid": uuid,
                    }
                )

        return results

    @classmethod
    def gen_uuid(cls, url):
        def stripped_url(url):
            return urljoin(url, urlparse(url).path)

        def string_hash_96(input_data: str) -> str:
            return hashlib.sha224(bytes(input_data, encoding="utf-8")).hexdigest()[:24]

        url = url.replace("\t", "").replace(" ", "")
        try:
            return string_hash_96(url)
        except Exception as e:
            print(e)
            return None

    @classmethod
    def normalize_url(cls, url, target_uri, strip_param=False):
        try:
            full_url = urljoin(target_uri, url)
            if strip_param:
                full_url = full_url.split("?")[0]
            return full_url
        except Exception as e:  # eg ed2k://
            print(e)
            return None


def parse(in_json_dir, out_has_alt_title_json):
    import shutil

    data = []
    for fn in os.listdir(in_json_dir):
        if fn.endswith(".gz"):
            data.extend(
                WATParser().parse_wat(
                    os.path.join(in_json_dir, fn),
                    in_json_dir.split("/")[-1],
                    fn.split(".")[0],
                )
            )

    with open(out_has_alt_title_json, "w") as fw:
        json.dump(data, fw)


if __name__ == '__main__':
    import sys
    
    parse(sys.argv[1], sys.argv[2])

