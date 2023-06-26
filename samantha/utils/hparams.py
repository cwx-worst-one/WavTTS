from typing import Dict, Optional  # noqa


class DotDict(dict):
    r"""Dictionary that supports dot notation.

    Args:
        py_dict (Optional[Dict]): A python dict object,
         will be recursively converted to DotDict.

    Example:
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

    def __init__(self, py_dict: dict = None, **kwargs):
        super().__init__()

        if py_dict is None:
            py_dict = {}

        py_dict.update(kwargs)

        for key, value in py_dict.items():
            if isinstance(value, dict):
                value = DotDict(value)
            self[key] = value
