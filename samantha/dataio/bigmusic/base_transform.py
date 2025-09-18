from typing import Any, Dict, List, Optional, Union

from mariana.utils.audio.audio_logger import AudioLogger

from samantha.dataio.bigmusic.base_exception import MusicMetaError
from samantha.dataio.bigmusic.transforms.utils import *

logger = AudioLogger()


class MusicMetaRWTransform:
    def __init__(
        self,
        in_key: Optional[Union[str, list[str], tuple[str]]],
        out_key: Optional[str],
        optional_keys: Optional[Union[list[str], tuple[str]]] = None,
        allow_empty_in: bool = False,
        remove_in_key: bool = False,
        overwrite_out_key: bool = True,
        disable_dot_ref: bool = False,
        overwrite_item: bool = False,
        maybe_return_list_output: bool = False,
        return_item_if_in_key_not_found: bool = False,
        bypass_condition: Optional[str] = None,
    ):
        """
        A simple transform that takes the value from in_key and put it into out_key.
        Args:
            in_key: key of the input item, or a list/tuple of in_keys (tuple is more recommended in case of in-place \
                modification), or None to take the entire input item.
            out_key: key of the output item, or None to not modify the item.
            optional_keys: when in_key is a list or tuple, the optional_keys are the keys that are optional. \
                If any of the optional_keys can't be find in the meta, the style tag of that category will be \
                an empty tag [""]. You might want to set allow_empty_in to True to allow empty input.
            allow_empty_in: allow empty input (in_key not found)
            overwrite_out_key: allow to overwrite the out_key if it exists
            remove_in_key: remove the in_key after the transformation
            disable_dot_ref: disable dot reference for in_key and out_key
            overwrite_item: overwrite the entire item by the returned value of the `call` method if out_key is None
            maybe_return_list_output: this flag shows the return value of the `call` method is a list, it is a \
                indicatior for following item_tranforms to do item by item
            return_item_if_in_key_not_found: return the item if in_key is not found in the item
        """
        self.in_key = in_key
        self.out_key = out_key
        self.optional_keys = optional_keys
        self.allow_empty_in = allow_empty_in
        self.remove_in_key = remove_in_key
        self.overwrite_out_key = overwrite_out_key
        self.disable_dot_ref = disable_dot_ref
        self.overwrite_item = overwrite_item
        self.maybe_return_list_output = maybe_return_list_output
        self.return_item_if_in_key_not_found = return_item_if_in_key_not_found
        self.bypass_condition = bypass_condition

        if self.overwrite_item:
            assert (
                self.out_key is None
            ), "out_key should always be None when overwrite_item is True"

        if self.bypass_condition:
            assert "==" in self.bypass_condition or "!=" in self.bypass_condition

    def __call__(self, item: dict, **kwargs):
        if item is None:
            return None

        if self.return_item_if_in_key_not_found:
            if not validate_item_with_keys(self.in_key, item):
                return item
            if not validate_item_with_keys("uttid", item):
                return item

        if self.bypass_condition is not None:
            sep = "!=" if "!=" in self.bypass_condition else "=="
            key, value = self.bypass_condition.split(sep)
            if sep == "!=":
                if str(item.get(key)) != value:
                    return item
            else:
                if str(item.get(key)) == value:
                    return item

        uttid = item.get("uttid")
        try:
            inner_item = self._take(item, uttid)

            if "uttid" not in kwargs:
                kwargs["uttid"] = uttid

            validate_item_with_optional_keys(
                self.in_key, self.optional_keys, inner_item
            )
            result = self.call(inner_item, **kwargs)
            if self.overwrite_item and self.out_key is None:
                return result
            return self._put(item, result, uttid)
        except MusicMetaError as e:
            logger.debug(f"{self._get_err_log_prefix(uttid)} {str(e)} ")
            return None

    def _get_log_prefix(self, uttid):
        return f"[uttid: {uttid}][transform: {self.__class__.__name__}]"

    def _get_err_log_prefix(self, uttid):
        return self._get_log_prefix(uttid) + f"[{err_loc()}]"

    def _take_one(
        self, item: Optional[dict], in_key: str, uttid: Optional[str]
    ) -> Optional[Any]:
        if in_key is None:
            return item
        keys = [in_key] if self.disable_dot_ref else in_key.split(".")
        for idx, key in enumerate(keys):
            if key not in item:
                if not self.allow_empty_in:
                    raise MusicMetaError(f'in_key "{in_key}" is not found in item')
                else:
                    return None
            if self.remove_in_key and idx == len(keys) - 1:
                item = item.pop(key)
            else:
                item = item[key]
        return item

    def _take(self, item: Optional[dict], uttid: Optional[str]) -> Optional[Any]:
        if isinstance(self.in_key, str):
            return self._take_one(item, self.in_key, uttid)
        items = [self._take_one(item, in_key, uttid) for in_key in self.in_key]
        return items

    def _put(self, item: dict, value: Any, uttid: Optional[str]) -> dict:
        if self.out_key is None:
            return item

        if isinstance(self.out_key, list):
            for k, v in zip(self.out_key, value):
                self._put_one(item, k, v, uttid)
        else:
            self._put_one(item, self.out_key, value, uttid)
        return item

    def _put_one(self, item: dict, key: Any, value: Any, uttid: Optional[str]) -> dict:
        keys = [key] if self.disable_dot_ref else key.split(".")
        inner_item = item
        for idx, k in enumerate(keys):
            if not isinstance(inner_item, dict):
                raise MusicMetaError(f'out_key "{key}" is not a dict')
            elif idx == len(keys) - 1:
                if not self.overwrite_out_key and k in inner_item:
                    raise MusicMetaError(f'out_key "{key}" already exists in item')
                inner_item[k] = value
            elif k not in inner_item:
                inner_item[k] = {}
                inner_item = inner_item[k]
            else:
                inner_item = inner_item[k]
        return item

    def call(self, item, **kwargs):
        """
        The item is not None unless self.allow_empty_in is True. If self.in_key is a list/tuple,
        the item will be a list of values.
        """
        return item
