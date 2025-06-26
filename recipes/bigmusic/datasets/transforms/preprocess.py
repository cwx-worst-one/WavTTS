import os
from typing import Any, Callable

from recipes.bigmusic.utils.common_utils import ExternalModule


class PreprocessConfig:
    PREPROCESS_REPO_DIR = os.getenv("PREPROCESS_DIR", "/opt/tiger/samantha/preprocess_py")
    PREPROCESS_CONFIG_PATH = os.getenv("PREPROCESS_CONFIG_PATH")


class PreprocessModule:
    def __init__(self):
        self.api = ExternalModule(PreprocessConfig.PREPROCESS_REPO_DIR, "preprocess.bigmusic.api")

    @property
    def process_lyrics(self) -> Callable[..., dict]:
        return self.api.getattr("process_lyrics")

    @property
    def process_tags(self) -> Callable[..., dict]:
        return self.api.getattr("process_tags")

    @property
    def process_tags_inst(self) -> Callable[..., dict]:
        return self.api.getattr("process_tags_inst")
    
    @property
    def generate_duration(self) -> Callable[..., int]:
        return self.api.getattr("generate_duration")

    @property
    def transform_duration(self) -> Callable[..., int]:
        return self.api.getattr("transform_duration")

    @property
    def transform_duration_inst(self) -> Callable[..., tuple[float, int]]:
        return self.api.getattr("transform_duration_inst")

    @property
    def config(self):
        if not PreprocessConfig.PREPROCESS_CONFIG_PATH:
            return self.api.getattr("DEFAULT_BIGMUSIC_CONFIG")
        return self.api.getattr("BigMusicConfig").from_json(PreprocessConfig.PREPROCESS_CONFIG_PATH)