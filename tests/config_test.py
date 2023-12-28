# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import unittest

from configs import search_config


class TestMetaCLIPConfig(unittest.TestCase):
    def test_400m(self):
        for config_name in ["b32_400m", "b16_400m", "l14_400m"]:
            config = search_config(config_name)
            self.assertEqual(config.name, config_name) 
            assert config.output_dir.endswith(config_name)

    def test_2_5B(self):
        for config_name in ["b32_fullcc", "b16_fullcc", "l14_fullcc", "h14_fullcc"]:
            config = search_config(config_name)
            self.assertEqual(config.name, config_name) 
            assert config.output_dir.endswith(config_name)


if __name__ == "__main__":
    # python -m unittest tests.config_test
    unittest.main()
