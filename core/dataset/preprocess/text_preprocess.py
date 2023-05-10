# encoding=utf-8
'''
preprocess about label.
'''
# pylint:disable=too-many-lines,too-many-return-statements
import io
import os
import re
import pickle
import random
import uuid
from collections import Counter, defaultdict
from itertools import groupby
import numpy as np
import jieba
import torch
from .preprocess import PREPROCESS


@PREPROCESS.register_module()
class RandomTemplateTextBuilder:
    '''RandomTemplateTextBuilder'''

    def __init__(self, templates=[], out_key="text"):
        self.templates = templates
        self.out_key = out_key

    def __call__(self, item, **kwargs):
        '''build text in terms of template.'''

        text = random.choice(self.templates)
        selected_template = text
        template_keys = list(set(re.findall("\{\S+\}", text)))
        for key in template_keys:
            item_key = key.lstrip("{").rstrip("}")
            text = re.sub(key, item[item_key], text)
        item[self.out_key] = text
        item[self.out_key + "_template"] = selected_template
        return item


@PREPROCESS.register_module()
class CategorySamplingTemplateTextBuilder:
    '''CategorySamplingTemplateTextBuilder'''

    def __init__(self, templates=[], out_key="text"):
        self.templates = templates
        self.out_key = out_key

    def __call__(self, item, **kwargs):
        '''build text in terms of template.'''

        distribution = [t[0] for t in self.templates]
        index = np.random.multinomial(1, distribution).argmax()
        text = self.templates[index][1]

        selected_template = text
        template_keys = list(set(re.findall("\{\S+\}", text)))
        for key in template_keys:
            item_key = key.lstrip("{").rstrip("}")
            text = re.sub(key, item[item_key], text)
        item[self.out_key] = text
        item[self.out_key + "_template"] = selected_template
        return item


@PREPROCESS.register_module()
class MergeCode:
    '''MergeCode'''

    def __init__(self, in_key="text", out_key="text"):
        self.in_key = in_key
        self.out_key = out_key

    def __call__(self, item, **kwargs):
        '''do merge.'''

        if isinstance(item[self.in_key], list):
            merged = [k for k, _ in groupby(item[self.in_key])]
        elif isinstance(item[self.in_key], str):
            to_merge = item[self.in_key].strip().split()
            merged = [k for k, _ in groupby(to_merge)]
            merged = " ".join(merged)
        else:
            raise ValueError("Unknown format for merge.")
        item[self.out_key] = merged
        return item
