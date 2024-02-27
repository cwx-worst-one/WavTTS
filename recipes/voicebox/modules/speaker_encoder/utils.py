import json
import os
import re
from typing import Dict

import fsspec
import yaml
from coqpit import Coqpit
import importlib



def to_camel(text):
    text = text.capitalize()
    text = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)
    text = text.replace("Tts", "TTS")
    text = text.replace("vc", "VC")
    return text


def find_module(module_path: str, module_name: str) -> object:
    module_name = module_name.lower()
    print(module_path, module_name)
    module = importlib.import_module(module_path + "." + module_name)
    class_name = to_camel(module_name)
    print(module_name, class_name)
    return getattr(module, class_name)


def read_json_with_comments(json_path):
    """for backward compat."""
    # fallback to json
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments
    input_str = re.sub(r"\\\n", "", input_str)
    input_str = re.sub(r"//.*\n", "\n", input_str)
    data = json.loads(input_str)
    return data


def _process_model_name(config_dict: Dict) -> str:
    """Format the model name as expected. It is a band-aid for the old `vocoder` model names.

    Args:
        config_dict (Dict): A dictionary including the config fields.

    Returns:
        str: Formatted modelname.
    """
    model_name = config_dict["model"] if "model" in config_dict else config_dict["generator_model"]
    model_name = model_name.replace("_generator", "").replace("_discriminator", "")
    return model_name


def register_config(model_name: str) -> Coqpit:
    """Find the right config for the given model name.

    Args:
        model_name (str): Model name.

    Raises:
        ModuleNotFoundError: No matching config for the model name.

    Returns:
        Coqpit: config class.
    """
    config_class = None
    config_name = model_name + "_config"

    # TODO: fix this
    if model_name == "xtts":
        from TTS.tts.configs.xtts_config import XttsConfig

        config_class = XttsConfig
    path = "recipes.voicebox.modules.speaker_encoder.configs"
    config_class = find_module(path, config_name)
    return config_class


def load_config(config_path: str) -> Coqpit:
    """Import `json` or `yaml` files as TTS configs. First, load the input file as a `dict` and check the model name
    to find the corresponding Config class. Then initialize the Config.

    Args:
        config_path (str): path to the config file.

    Raises:
        TypeError: given config file has an unknown type.

    Returns:
        Coqpit: TTS config object.
    """
    config_dict = {}
    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with fsspec.open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        try:
            with fsspec.open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError:
            # backwards compat.
            data = read_json_with_comments(config_path)
    else:
        raise TypeError(f" [!] Unknown config file type {ext}")
    config_dict.update(data)
    model_name = _process_model_name(config_dict)
    config_class = register_config(model_name.lower())
    config = config_class()
    config.from_dict(config_dict)
    return config



