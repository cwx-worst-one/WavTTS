# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import os.path as osp
import shutil
import sys
import tempfile
from argparse import Action, ArgumentParser
from collections import abc
from importlib import import_module

from addict import Dict
from yapf.yapflib.yapf_api import FormatCode
from .fileio.io import load as fload, dump as fdump
from .path import check_file_exist
from .dist_hdfs import dist_hdfs_get

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
# config.py is located at DOLPHIN_BASE_DIR/core/utils/configs.py
DOLPHIN_BASE_DIR = osp.join(osp.dirname(__file__), "../..")


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def add_args(parser, cfg, prefix=''):
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + '.')
        elif isinstance(v, abc.Iterable):
            parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+')
        else:
            print(f'cannot parse key {prefix + k} of type {type(v)}')
    return parser


class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"

    """

    @staticmethod
    def _hdfsfile2dict(filename):
        # hdfs files should fetch to local, and all base config files.
        local_filename = filename.split('/')[-1]
        dist_hdfs_get(filename, f'{DOLPHIN_BASE_DIR}/configs/hdfs', local_filename)
        return Config._file2dict(f'{DOLPHIN_BASE_DIR}/configs/hdfs/{local_filename}')

    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        if filename.endswith('.py'):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                # pylint:disable=consider-using-with
                temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py')
                temp_config_name = osp.basename(temp_config_file.name)
                shutil.copyfile(filename, osp.join(temp_config_dir, temp_config_name))
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value for name, value in mod.__dict__.items() if not name.startswith('__')
                }
                # delete imported module
                del sys.modules[temp_module_name]
                # close temp file
                temp_config_file.close()
        elif filename.endswith(('.yml', '.yaml', '.json')):
            cfg_dict = fload(filename)
        else:
            raise IOError('Only py/yml/yaml/json type are supported now!')

        cfg_text = filename + '\n'
        with open(filename, 'r', encoding='utf-8') as f:
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.join(DOLPHIN_BASE_DIR, 'configs/_base_')
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(base_filename, list) else [base_filename]

            cfg_dict_list = []
            cfg_text_list = []
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError('Duplicate key is not allowed among bases')
                base_cfg_dict.update(c)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b):
        # merge dict `a` into dict `b` (non-inplace). values in `a` will
        # overwrite `b`.
        # copy first to avoid inplace modification
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b and not v.pop(DELETE_KEY, False):
                if not isinstance(b[k], dict):
                    raise TypeError(
                        f'{k}={v} in child config cannot inherit from base '
                        f'because {k} is a dict in the child config but is of '
                        f'type {type(b[k])} in base config. You may set '
                        f'`{DELETE_KEY}=True` to ignore the base config'
                    )
                b[k] = Config._merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    @staticmethod
    def fromfile(filename):
        if filename.startswith('hdfs'):
            cfg_dict, cfg_text = Config._hdfsfile2dict(filename)
        else:
            cfg_dict, cfg_text = Config._file2dict(filename)
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from config file automatically (experimental)"""
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f'cfg_dict must be a dict, but got {type(cfg_dict)}')

        super().__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super().__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = ''
        super().__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    @property
    def cfg_dict(self):
        return self._cfg_dict

    @property
    def pretty_text(self):

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),' for v_ in v
                ).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style='pep8',
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True,
        )
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def __repr__(self):
        return f'Config (path: {self.filename}): {self._cfg_dict.__repr__()}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def dump(self, file=None):
        cfg_dict = super().__getattribute__('_cfg_dict').to_dict()

        if file is None:
            self_format = self.filename.split('.')[-1]
            if self_format == 'py':
                return self.pretty_text
            return fdump(cfg_dict, file_format=self_format)

        file_format = file.split('.')[-1]
        if file_format == 'py':
            with open(file, 'w', encoding='utf-8') as f:
                f.write(self.pretty_text)
        else:
            fdump(cfg_dict, file)
        return None

    def merge_from_dict(self, options):
        """Merge dict into cfg_dict

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

        Args:
            options (dict): dict of configs to merge from.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super().__getattribute__('_cfg_dict')
        super().__setattr__('_cfg_dict', Config._merge_a_into_b(option_cfg_dict, cfg_dict))

    def merge_from_list(self, options):
        """Merge list into cfg_dict

        Merge the unknown argument list into this cfg.

        Examples:
            >>> options = ['model.backbone.depth', 50,
            ...            'model.backbone.with_cp', 'true']
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_list(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

        Args:
            options (list): list of configs to merge from.
        """
        # pylint:disable=too-many-branches
        if len(options) % 2 != 0:
            raise ValueError('The parameter format is wrong, there are an odd number of parameters')

        i = 0
        while i < len(options):
            full_key = options[i]
            if full_key.startswith('--'):
                full_key = full_key[2:]
                if full_key.count('-') != 0:
                    raise ValueError(
                        'The parameter format is wrong, {} contains \"-\"'.format(full_key)
                    )

            else:
                raise ValueError(
                    'The parameter format is wrong, {} does not start with \"--\"'.format(full_key)
                )
            i += 1

            val = None
            if i < len(options):
                val = options[i]
                i += 1
            if val is not None and val.lower() in ['true', 'false']:
                val = val.lower() == 'true'
            try:
                val = int(val)
            except Exception:
                try:
                    val = float(val)
                except Exception:
                    pass

            key_list = full_key.split('.')
            my_dict = self._cfg_dict
            for subkey in key_list[:-1]:
                if subkey.isdigit() and isinstance(my_dict, list):
                    my_dict = my_dict[int(subkey)]
                else:
                    my_dict.setdefault(subkey, ConfigDict())
                    my_dict = my_dict[subkey]

            subkey = key_list[-1]
            if subkey.isdigit() and isinstance(my_dict, list):
                subkey = int(subkey)
            my_dict[subkey] = val


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return val.lower() == 'true'
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(',')]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)
