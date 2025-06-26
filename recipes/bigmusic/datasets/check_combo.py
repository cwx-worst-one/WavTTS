# coding:utf-8

from recipes.bigmusic.datasets.utils.zh_vocab import (
    VOCAB2ID_SA,
    VOCAB2ID_AUDIO_V3,
    VOCAB2ID_MIX_V3,
)

from recipes.bigmusic.datasets.combo_audio_tags import (
    COMBO_GENRE_TO_MULTITAG_V4,
)

from recipes.bigmusic.datasets.mir_data_util import (
    rewrite_style_input_to_multi_tag_combo_v4,
)

def check_combo_validity(vocab2id_map, multitag_map):
    vocab2id = vocab2id_map.to_dict()
    for key in multitag_map:
        for item in multitag_map[key]:
            for _item in item.split('|'):
                for v in _item.split(','):
                    if v and v not in vocab2id:
                        print('%s not in vocab2id' % v)

if __name__ == '__main__':
    import sys
    import codecs
    import csv
    input_file = sys.argv[1]

    vocab2id_map = VOCAB2ID_MIX_V3
    multitag_map = COMBO_GENRE_TO_MULTITAG_V4
    check_combo_validity(vocab2id_map, multitag_map)
    with codecs.open(input_file, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            text_prompt = row['text_prompt']
            _ = rewrite_style_input_to_multi_tag_combo_v4(text_prompt)




