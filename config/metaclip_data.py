# Copyright (c) Meta Platforms, Inc. and affiliates


from dataclasses import dataclass


@dataclass
class metaclip_400m:
    metadata = "metadata.json"
    shard_dir = "data/metaclip_400m/shard"
    index_dir = "data/metaclip_400m/index"
    start_shard = 0
    end_shard = 60800
    max_match = 20000  # the magic 20k t


@dataclass
class metaclip_2_5b:
    metadata = "metadata.json"
    shard_dir = "data/metaclip_2_5b/shard"
    index_dir = "data/metaclip_2_5b/index"
    start_shard = 0
    end_shard = 200000
    max_match = 170000

