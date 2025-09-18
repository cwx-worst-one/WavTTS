import random

import yaml

from samantha.dataio.bigmusic.base_transform import MusicMetaRWTransform
from samantha.dataio.bigmusic.transforms.utils import *


class PromptSelector(MusicMetaRWTransform):

    def __init__(
        self,
        in_key: list = [
            "keywords",
            "keywords_largedrop",
            "descriptions",
            "web_description",
        ],
        out_key: str = "prompt",
        config_path: str = "prompt_process.yaml",
        result_field: str = "result",
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=True, **kwargs)
        self.config = yaml.safe_load(open(config_path))
        self.result_field = result_field

    def call(self, item: dict, **kwargs) -> dict:
        keywords, keywords_largedrop, descriptions, web_description = item

        # put everything in a dict
        feat = {
            "keywords": keywords,
            "keywords_largedrop": keywords_largedrop,
            "web_description": web_description,
        }
        feat.update(descriptions)

        aggregate_configs = self.config["aggregates"]
        for aggregate_config in aggregate_configs:
            aggregate_feature(feat, aggregate_config)

        prompt = feat[self.result_field]
        if isinstance(prompt, list):
            prompt = prompt[0]

        return prompt
