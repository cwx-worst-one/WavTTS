'''
backup a key.
'''
import copy
from .preprocess import PREPROCESS


@PREPROCESS.register_module()
class Backup:
    '''backup some keys before been processed'''

    def __init__(self, key=None):
        '''init
        Args:
            key(str): a key for buckup
        '''
        self.key = key

    def __call__(self, item_data, **_kwargs):
        '''do item backup
        Args:
            item_data(dict): input data.

        Returns:
            dict: data with key backup
        '''
        if item_data is None or self.key is None:
            return item_data

        item_data[self.key + '_bak'] = copy.deepcopy(item_data[self.key])
        return item_data
