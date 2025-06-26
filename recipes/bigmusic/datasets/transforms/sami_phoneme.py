import os
import contextlib
import logging
from functools import lru_cache

from recipes.bigmusic.utils.common_utils import record_time
from recipes.datasets.mcc.fetch_remote_frontend_res import remote_run as _remote_run

from ..utils.sami_parser import convert_phonemes as _convert_phonemes


logger = logging.getLogger(__file__)

try:
    from sami_tts_api.engine import TtsEngine, generate_tts_config
except Exception as e:
    logger.info(f"Failed loading sami_tts_api: {e}")


class SamiTextToPhonemeTransform:
    def __init__(
        self,
        lib_path: str = "/opt/tiger/sami_engine_cleaned/libs/libsami.so",
        fe: str = "/opt/tiger/sami_tts_api/models/tts_chinese_frontend_model__42.0.model",
        fe_task: str = "tts_chinese_frontend_model",
    ) -> None:
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")  # independent logger for each instance
        model_name = os.environ.get("FRONTEND_MODEL_NAME", "").strip()
        if model_name and os.getenv('USE_LOCAL_TTS_FRONTEND') == '1':
            fe = f"/opt/tiger/sami_tts_api/models/{model_name}" if model_name else fe
            self.cfg = generate_tts_config()
            self.engine = TtsEngine(lib_path=lib_path, fe=fe)
            self.ex = self.engine.create_fe_executor(task_type=fe_task)
            self.logger.info("Using local tts front-end results")
        else:
            self.cfg = None
            self.engine = None
            self.ex = None
            self.logger.info("Using cloud tts front-end results")

    def __call__(self, text: str, lang: str="zh_en"):
        record = {}
        if self.ex is None:
            with record_time(record, "remote_tts_frontend"):
                phonemes = _remote_run(text, lang=lang)
        else:
            with record_time(record, "local_tts_frontend"):
                with contextlib.redirect_stdout(None):
                    phonemes = self.ex.run(text, config=self.cfg)[0]
        self.logger.info(f"{self.__class__.__name__}.__call__ execution time: {record}")
        return phonemes

    @classmethod
    @lru_cache
    def init_cached(
        cls,
        lib_path: str = "/opt/tiger/sami_engine_cleaned/libs/libsami.so",
        fe: str = "/opt/tiger/sami_tts_api/models/tts_chinese_frontend_model__42.0.model",
        fe_task: str = "tts_chinese_frontend_model",
    ) -> "SamiTextToPhonemeTransform":
        return cls(lib_path, fe, fe_task)


def transform_phonemes_by_language(phonemes: str, language: str) -> str:
    return _convert_phonemes(phonemes, language)
