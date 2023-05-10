''' enhanced dict '''


class FalconDict(dict):
    '''
    The usage method is similar to ordinary dict,
    adding the function of falcondict.xxx and the function of judging conflicts during update
    '''

    def update(self, other, force=False):
        if not force:
            for key in self.keys() & other.keys():
                if self[key] != other[key]:
                    raise Exception("two dicts have conflicting fields.")
        super().update(other)

    def __add__(self, other):
        '''merge dict with force is False'''
        origin = self.copy()
        origin.update(other)
        return origin

    def __getattr__(self, key):
        '''getattr'''
        value = self.get(key, None)
        return self.__class__(value) if isinstance(value, dict) else value

    def __setattr__(self, key, value):
        '''setattr'''
        self[key] = value

    def key_list(self):
        '''key_list'''
        return list(self.keys())

    def value_list(self):
        '''value list'''
        return list(self.values())

    def item_list(self):
        '''to list'''
        return list(self.items())

    def copy(self):
        'shallow copy'
        return self.__class__(self)
