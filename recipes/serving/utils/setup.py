from hyperpyyaml import load_hyperpyyaml

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


def setup_app(app: str, handler: Handler = DefaultHandler):
    init_logging_config()
    
    with open(f"recipes/serving/apps/{app}/config.yaml", "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f)

    configs = DotDict(configs)

    func_path_str = configs.repo.func_path
    func_path = importlib.import_module(func_path_str)

    preload_models = getattr(func_path, configs.repo.get('preload_model_func_name', 'preload_models'))
    api_main = getattr(func_path, configs.repo.get('api_main_func_name', 'api_main'))

    handler.preload_models = preload_models
    handler.api_main = api_main
    handler.inputs = configs.inputs
    handler.outputs = configs.outputs
    handler.app_name = app

    return api_main, preload_models
