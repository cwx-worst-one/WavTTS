'''
Edit keys in items
'''

from .preprocess import PREPROCESS


@PREPROCESS.register_module()
class AddKey:
    '''
    Add keys from the kwargs to the item
    '''

    def __init__(self, kwargs_key='uttid', item_key='uttid'):
        '''init.'''
        self.kwargs_key = kwargs_key
        self.item_key = item_key

    def __call__(self, item, **kwargs):
        '''
        do call.
        '''
        if item is None or self.kwargs_key not in kwargs:
            return None
        item[self.item_key] = kwargs.pop(self.kwargs_key, None)
        return item
