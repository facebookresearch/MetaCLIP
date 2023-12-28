# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import unittest
import json

from dataclasses import dataclass

from configs import search_config


@dataclass
class metaclip_test:
    metadata = "metadata.json"
    shard_dir = "data/metaclip_test/shard"
    index_dir = "data/metaclip_test/index"
    start_shard = 0
    end_shard = 1
    max_match = 20000  # the magic 20k t


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.args = metaclip_test()
        self.texts_iter = lambda x: ["a dog", "a cat"]

    def test_data_config(self):
        for config_name in ["metaclip_400m", "metaclip_2_5b"]:
            config = search_config(config_name)

    def test_all(self):
        self._substr_indexing()
        self._entry_count()
        self._balance_sampling()


    def _substr_indexing(self):
        from metaclip.indexing.substr_indexing import build_shards_index
        args = self.args

        with open(args.metadata) as f:
            metadata = json.load(f)

        build_shards_index(args.index_dir, metadata, self.texts_iter, args.start_shard, args.end_shard)

    def _entry_count(self):
        from metaclip.indexing.entry_count import entry_count
        entry_count(self.args)

    def _balance_sampling(self):
        from metaclip.indexing.balance_sampling import build_subset_index
        build_subset_index(self.args)


if __name__ == "__main__":
    # python -m unittest tests.pipeline_test
    unittest.main()
