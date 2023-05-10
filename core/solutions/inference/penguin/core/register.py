import importlib
from absl import logging


class Register:
    """Module register"""

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, param):
        """Decorator to register a function or class."""

        def decorator(key, value):
            self[key] = value
            return value

        if callable(param):
            # @reg.register
            return decorator(None, param)
        # @reg.register('alias')

        return lambda x: decorator(param, x)

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as ex:
            logging.error(f"module {key} not found: {ex}")
            raise ex

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()


class Registers:
    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    model = Register('model')
    inference = Register("inference")
    processor = Register('processor')


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        logging.warning("Module {} import failed: {}".format(name, err))


def import_all_modules_for_register(custom_module_paths=None):
    """Import all modules for register."""
    modules = [
        'core.solutions.inference.penguin.core.processor.jointer_small_processor',
        'core.solutions.inference.penguin.core.processor.decoder_processor',
        'core.solutions.inference.penguin.core.processor.encoder_processor',
        'core.solutions.inference.penguin.core.processor.jointer_processor',
        'core.solutions.inference.penguin.core.processor.predictor_processor',
        'core.solutions.inference.penguin.core.processor.get_result_processor',
        'core.solutions.inference.penguin.core.processor.input_generator_processor',
        'core.solutions.inference.penguin.core.processor.feature_input_processor',
        'core.solutions.inference.penguin.core.processor.nnlm_processor',
        'core.solutions.inference.penguin.core.processor.cif.cif_decoder_processor',
        'core.solutions.inference.penguin.core.processor.cif.cif_get_result_processor',
    ]
    #   for base_dir, modules in ALL_MODULES:
    #        for name in modules:
    #            full_name = base_dir + "." + name
    #            modules.append(full_name)
    if isinstance(custom_module_paths, list):
        modules += custom_module_paths
    errors = []
    for module in modules:
        try:
            print(module)
            importlib.import_module(module)
        except ImportError as error:
            errors.append((module, error))
    _handle_errors(errors)
