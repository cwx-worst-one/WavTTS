''' test label preprocessing '''
from core.dataset.preprocess import (
    PunctuationFilter,
)


if __name__ == "__main__":
    punct_filter = PunctuationFilter(key='label')
    item_data = {}

    item_data['label'] = "test string, format."
    print("before PunctuationFilter: {}".format(item_data['label']))
    punct_filter(item_data)
    print("after PunctuationFilter: {}".format(item_data['label']))

    item_data['label'] = ["test", "list,", "format."]
    print("before PunctuationFilter: {}".format(item_data['label']))
    punct_filter(item_data)
    print("after PunctuationFilter: {}".format(item_data['label']))

    item_data['label'] = ["test", "list", ",", "format", "."]
    print("before PunctuationFilter: {}".format(item_data['label']))
    punct_filter(item_data)
    print("after PunctuationFilter: {}".format(item_data['label']))
