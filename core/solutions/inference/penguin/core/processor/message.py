import re
from absl import logging
import pickle

_NAME_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_SQUARE_BRACKET_PATTERN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]')


class Message(dict):
    """A simple helper to maintain a dict.
    It is a sub-class of dict with the following extensions/restrictions:
      - It supports attr access to its members (see examples below).
      - Member keys have to be valid identifiers.
    E.g.::
        >>> foo = Message()
        >>> foo['x'] = 10
        >>> foo.y = 20
        >>> assert foo.x * 2 == foo.y
    """

    # Disable pytype attribute checking.
    _HAS_DYNAMIC_ATTRIBUTES = True
    # keys in this list are not allowed in a Message.
    _RESERVED_KEYS = frozenset(dir(dict))
    # sentinel value for deleting keys used in Filter.
    _DELETE = object()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in self.keys():
            assert isinstance(key, str), (
                'Key in a Message has to be a six.string_types.',
                'Currently type: %s,' ' value: %s' % (str(type(key)), str(key)),
            )
            Message.check_key(key)
            assert key not in Message._RESERVED_KEYS, '%s is a reserved key' % key

    def __setitem__(self, key, value):
        # Make sure key is a valid expression and is not one of the reserved
        # attributes.
        assert isinstance(key, str), (
            'Key in a Message has to be a six.string_types.',
            'Currently type: %s,' 'value: %s' % (str(type(key)), str(key)),
        )
        Message.check_key(key)
        assert key not in Message._RESERVED_KEYS, '%s is a reserved key' % key
        super().__setitem__(key, value)

    def __setattr__(self, name, value):
        self.__setitem__(name, value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError('%s; available attributes: %s' % (err, sorted(list(self.keys()))))

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as err:
            raise AttributeError('%s; available attributes: %s' % (err, sorted(list(self.keys()))))

    def copy(self):  # Don't delegate w/ super: dict.copy() -> dict.
        return Message(self)

    def __deepcopy__(self, unused_memo):
        """Deep-copies the structure but not the leaf objects."""
        return self.deepcopy()

    def deepcopy(self):
        """Deep-copies the structure but not the leaf objects."""
        return self.pack(self.flatten())

    @staticmethod
    def from_nested_dict(msg):
        """Converts every dict in nested structure 'msg' to a Message."""
        if isinstance(msg, dict):
            res = Message()
            for key, val in msg.items():
                res[key] = Message.from_nested_dict(val)
            return res
        elif isinstance(msg, (list, tuple)):
            return type(msg)(Message.from_nested_dict(val) for val in msg)
        else:
            return msg

    @staticmethod
    def check_key(key):
        """Asserts that key is valid Message key.
        key is a valid variable name or a wav file name"""
        if not (isinstance(key, str) and _NAME_PATTERN.match(key) or key.endswith('.wav')):
            raise ValueError('Invalid Message key \'{}\''.format(key))

    @staticmethod
    def square_bracket_index(key):
        """Extracts the name and the index from the indexed key (e.g., k[0])."""
        _match = _SQUARE_BRACKET_PATTERN.fullmatch(key)
        if not _match:
            return key, None
        else:
            return _match.groups()[0], int(_match.groups()[1])

    def get_item(self, key):
        """Gets the value for the nested `key`.
        Note that indexing lists is not supported, names with underscores will be
        considered as one key.
        Args:
          key: str of the form
            `([A-Za-z_][A-Za-z0-9_]*)(.[A-Za-z_][A-Za-z0-9_]*)*.`.
        Returns:
          The value for the given nested key.
        Raises:
          KeyError: if a key is not present.
          IndexError: when an intermediate item is a list and we try to access
            an element which is out of range.
          TypeError: when an intermediate item is a list and we try to access
            an element of it with a string.
        """
        current = self
        for k in key.split('.'):
            k, idx = self.square_bracket_index(k)
            current = current[k]
            if idx is not None:
                current = current[idx]
        return current

    def get(self, key, default=None):
        """Gets the value for nested `key`, returns `default`
        if key does not exist. Note that indexing lists is not supported,
        names with underscores will beconsidered as one key.
        Args:
          key: str of the form
            `([A-Za-z_][A-Za-z0-9_]*)(.[A-Za-z_][A-Za-z0-9_]*)*.`.
          default: Optional default value, defaults to None.
        Returns:
          The value for the given nested key or `default` if the key does not exist.
        """
        try:
            return self.get_item(key)
        except (KeyError, IndexError, TypeError):
            return default

    def set(self, key, value):
        r"""Sets the value for a nested key.
        There is limited support for indexing lists when square bracket indexing is
        used, e.g., key[0], key[1], etc. Names with underscores will be considered
        as one key. When key[idx] is set, all of the values with indices before idx
        must be already set. E.g., setting key='a[2]' to value=42 when
        key='a' wasn't referenced before will throw a ValueError. Setting key='a[0]'
        will not.
        Args:
          key: str of the form key_part1.key_part2...key_partN where each key_part
            is of the form `[A-Za-z_][A-Za-z0-9_]*` or
            `[A-Za-z_][A-Za-z0-9_]*\[\d+\]`
          value: The value to insert.
        Raises:
          ValueError if a sub key is not a Message or dict or idx > list length
          for key='key[idx]'.
        """
        current = self
        sub_keys = key.split('.')
        for i, k in enumerate(sub_keys):
            self.check_key(k)  # check_key allows k to be of form k[\d+]
            k, idx = self.square_bracket_index(k)
            # this is key with index pointing to a list item.
            if idx is not None:
                # create a list if not there yet.
                if k not in current:
                    current[k] = []
                if idx > len(current[k]):
                    raise ValueError(
                        'Error while setting key {}. The value under {} is a'
                        ' list and the index {} is greater than the len={} '
                        'of this list'.format(key, k, idx, len(current[k]))
                    )
                elif idx == len(current[k]):
                    current[k].extend([None])  # this None will be overwritten right away.

            # We have reached the terminal node, set the value.
            if i == (len(sub_keys) - 1):
                if idx is None:
                    current[k] = value
                else:
                    current[k][idx] = value
            else:
                if idx is None:
                    if k not in current:
                        current[k] = Message()
                    current = current[k]
                else:
                    if current[k][idx] is None:
                        current[k][idx] = Message()
                    current = current[k][idx]
                if not isinstance(current, (dict, Message)):
                    raise ValueError(
                        'Error while setting key {}. Sub key "{}" is of type'
                        ' {} but must be a dict or Message.'
                        ''.format(key, k, type(current))
                    )

    def _recursivemap(self, fun, flatten=False):
        """Traverse recursively into lists, dicts, and Messages applying `fun`.
        Args:
          fun: The funtion to apply to each item (leaf node).
          flatten: If true, the result should be a single flat list. Otherwise the
            result will have the same structure as this Message.
        Returns:
          The result of applying fun.
        """

        def recurse(vec, key=''):
            """Helper funtion for _recursivemap."""
            if isinstance(vec, dict):
                ret = [] if flatten else type(vec)()
                deleted = False
                for k in sorted(vec.keys()):
                    res = recurse(vec[k], key + '.' + k if key else k)
                    if res is self._DELETE:
                        deleted = True
                        continue
                    elif flatten:
                        ret += res
                    else:
                        ret[k] = res
                if not ret and deleted:
                    return self._DELETE
                return ret
            elif isinstance(vec, list):
                ret = []
                deleted = False
                for i, ele in enumerate(vec):
                    res = recurse(ele, '%s[%d]' % (key, i))
                    if res is self._DELETE:
                        deleted = True
                        continue
                    elif flatten:
                        ret += res
                    else:
                        ret.append(res)
                if not ret and deleted:
                    return self._DELETE
                return ret
            else:
                ret = fun(key, vec)
                if flatten:
                    ret = [ret]
                return ret

        res = recurse(self)
        if res is self._DELETE:
            return [] if flatten else Message()
        return res

    def flatten(self):
        """Returns a list containing the flattened values in the `.Message`.
        Unlike py_utils.flatten(), this will only descend into lists, dicts, and
        Messages and not tuples, or namedtuples.
        """
        return self._recursivemap(lambda _, v: v, flatten=True)

    def flatten_items(self):
        """flatten the `.Message` and returns <key, value> pairs in a list.
        Returns:
          A list of <key, value> pairs, where keys for nested entries will be
          represented in the form of `foo.bar[10].baz`.
        """
        return self._recursivemap(lambda k, v: (k, v), flatten=True)

    def pack(self, lst):
        """Returns a copy of this with each value replaced by a value in lst."""
        assert len(self.flatten_items()) == len(lst)
        v_iter = iter(lst)
        return self._recursivemap(lambda unused_k, unused_v: next(v_iter))

    def transform(self, fun):
        """Returns a copy of this `.Message` with fun applied on each value."""
        return self._recursivemap(lambda _, v: fun(v))

    def transform_withkey(self, fun):
        """Returns a copy of this `.Message` with fun applied on each keyvalue."""
        return self._recursivemap(fun)

    def is_compatible(self, other):
        """Returns true if self and other are compatible.
        If x and y are two compatible `.Message`, `x.Pack(y.flatten())` produces y
        and vice versa.
        Args:
          other: Another `.Message`.
        """
        items = self._recursivemap(lambda k, _: k, flatten=True)
        other_items = other._recursivemap(
            lambda k, _: k, flatten=True
        )  # pylint: disable=protected-access
        return items == other_items

    def filters(self, fun):
        """Returns a copy with entries where fun(entry) is True."""
        return self.filer_keyval(lambda _, v: fun(v))

    def filer_keyval(self, fun):
        """Returns a copy of this `.Message` filtered by fun.
        If fun(key, entry) is True, the entry is copied into the returned Message.
        Otherwise, it is not copied.
        Args:
          fun: a callable of (string, entry)->boolean.
        Returns:
          A `.Message` contains copied entries from this `'.Message`.
        """
        return self._recursivemap(lambda k, v: v if fun(k, v) else self._DELETE)

    def _tostrings(self):
        """Returns debug strings in a list for this `.Message`."""
        its = self.flatten_items()
        mlen = max([len(k) for k, _ in its]) if its else 0
        return sorted([k + ' ' * (4 + mlen - len(k)) + str(v) for k, v in its])

    def debug_string(self):
        """Returns a debug string for this `.Message`."""
        return '\n'.join(self._tostrings())

    def vlog(self, level=None, prefix=None):
        """Logs the debug string at the level."""
        if level is None:
            level = 0
        if prefix is None:
            prefix = 'nmap: '
        for strs in self._tostrings():
            logging.vlog(level, '%s %s', prefix, strs)

    def __dir__(self):
        """dir() that includes flattened keys in returned output."""
        keys = self._recursivemap(lambda k, v: k, flatten=True)
        return keys + super().__dir__()

    def serialization(self):
        return pickle.dumps(self)

    @staticmethod
    def deserialization(message):
        return pickle.loads(message)
