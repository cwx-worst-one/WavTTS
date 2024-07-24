from hyperpyyaml import load_hyperpyyaml

import logging
import importlib
from recipes.serving.utils.py_logging import init_logging_config
from recipes.serving.handler import Handler
from recipes.serving.handler.default import DefaultHandler


class DotDict(dict):
    """Dictionary that supports dot notation.

    Arguments
    --------
    py_dict: dict, {}
        A python dict object, will be recursively converted to DotDict.

    Example
    --------
    >>> d = {"key1": "val1", "key2": {"key3": "val3"}}
    >>> dot_d = DotDict(d)
    >>> dot_d.key1
    'val1'
    >>> dot_d["key1"]
    'val1'
    >>> dot_d.key2.key3
    'val3'
    >>> dot_d["key2"]["key3"]
    'val3'
    >>> dot_d.key2.key3 = "new_val"
    >>> dot_d.key2.key3
    'new_val'
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, py_dict: dict = {}):
        for key, value in py_dict.items():
            if isinstance(value, list):
                for i in range(len(value)):
                    if isinstance(value[i], dict):
                        value[i] = DotDict(value[i])
            if isinstance(value, dict):
                value = DotDict(value)
            self[key] = value


def get_model_configs(app):
    model_config_path = ''
    if app == 'BigTTS':
        model_config_path = 'recipes/text2semantic/conf/llama/inference_wvae_icl_lang_spk_tag_deploy.yaml'
    elif app == 'Lyrics2Song':
        model_config_path = 'recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m_deploy.yaml'
    elif app == 'Instrumental':
        model_config_path = 'recipes/bigmusic/conf/Q4_2023/inference/inference_instrumental_sstk_v8_deploy.yaml'
    elif app == "Research":
        model_config_path = 'recipes/research/diff/conf/prod/instrumental.yaml'
    else:
        raise ValueError(f"Unknown app: {app}")

    logging.info(f"model_config_path: {model_config_path}")
    with open(model_config_path, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f)
    return DotDict(configs)


def get_server_configs(app):
    with open(f"recipes/serving/apps/{app}/config.yaml", "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f)
    return DotDict(configs)


def setup_app(app: str, handler: Handler = DefaultHandler):
    init_logging_config()
    logging.info("***** setup app *****")

    configs = get_server_configs(app)

    func_path_str = configs.repo.func_path
    func_path = importlib.import_module(func_path_str)

    preload_models = getattr(func_path, configs.repo.get('preload_model_func_name', 'preload_models'))
    api_main = getattr(func_path, configs.repo.get('api_main_func_name', 'api_main'))

    handler.preload_models = preload_models
    handler.api_main = api_main
    handler.inputs = configs.inputs
    handler.outputs = configs.outputs
    handler.app_name = app

    logging.info("***** setup app success *****")
    return api_main, preload_models
