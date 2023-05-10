# pylint: disable=missing-module-docstring,missing-function-docstring
from abc import ABCMeta, abstractmethod


class BaseFileHandler(metaclass=ABCMeta):
    '''BaseFileHandler.'''

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath, mode='r', **kwargs):
        with open(filepath, mode, encoding='utf-8') as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, mode='w', **kwargs):
        with open(filepath, mode, encoding='utf-8') as f:
            self.dump_to_fileobj(obj, f, **kwargs)
