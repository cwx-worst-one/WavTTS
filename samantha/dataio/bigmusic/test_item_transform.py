import json
import logging
import random
import unittest

import numpy as np
import yaml

from samantha.dataio.bigmusic.bigmusic_compose import *

logging.basicConfig(level=logging.DEBUG)
random.seed(123)


# TESTING PURPOSE
def make_data():
    test_meta_json = ".tests/unittests/dataio/assets/test_temp_meta.json"
    # hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/rui.xia/bigmusic/misc/test_audio.wav
    test_wav = "tests/unittests/dataio/assets/test_audio.wav"

    data = {"uttid": "test_sample"}

    with open(test_meta_json, "r") as fp:
        test_meta = json.load(fp)

    data["audio"] = open(test_wav, "rb").read()
    data["meta"] = json.dumps(test_meta)
    return data


def make_data_converted():
    data = make_data()
    convert_meta_to_dict = ConvertMetaToDict()
    data = convert_meta_to_dict(data)
    return data


def make_eval_data():
    test_meta_json = "tests/unittests/dataio/assets/test_eval_meta.json"
    data = {"uttid": "test_sample"}

    with open(test_meta_json, "r") as fp:
        test_meta = json.load(fp)

    data["meta"] = json.dumps(test_meta)

    return data


def make_data_token():

    umm_token_file = (
        "tests/unittests/dataio/assets/umm_66eb342e-fec5-48f9-9263-d457d05cd5b1.bin"
    )
    test_meta = "tests/unittests/dataio/assets/umm_66eb342e-fec5-48f9-9263-d457d05cd5b1.meta.json"

    with open(test_meta, "r") as fp:
        test_meta = json.load(fp)

    umm_token = pickle.load(open(umm_token_file, "rb"))

    data = {
        "uttid": "test_token_sample",
        "meta": json.dumps(test_meta),
        "umm_token": pickle.dumps({"umm_token": umm_token}),
    }

    return data


# single test single item
def test_UtteranceParser():

    lyrics_transform = UtteranceParser(out_key="lyrics")

    sample_data = make_data_converted()

    sample_data = lyrics_transform(sample_data)
    print(sample_data["lyrics"])


def test_structure_parser():
    sample_data = make_data_converted()
    transform = StructureParser()
    sample_data = transform(sample_data)
    print(sample_data[transform.out_key])


def test_style_tag_parser_and_freeform_parser():
    sample_data = make_data_converted()
    transform = StyleTagParser()
    sample_data = transform(sample_data)
    print(sample_data[transform.out_key])

    transform = FreeformTextParser()
    sample_data = transform(sample_data)
    print(sample_data[transform.out_key])


# mulple test
def write_json(d, fp):
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def test_all_transforms():

    cfg_path = "tests/unittests/dataio/assets/v5_dataloader_mariana.yaml"
    with open(cfg_path) as f:
        kwargs = yaml.full_load(f)
    print(kwargs)
    transform_list = kwargs["data"]["train_item_transform"]
    print(transform_list)

    train_item_transform = build_item_augmentation(transform_list)

    test_init_dict = make_data()
    test_processed_dict = train_item_transform(test_init_dict)
    print(type(test_processed_dict))

    print(type(test_processed_dict[0]))
    print(test_processed_dict[0].keys())

    for key, val in test_processed_dict[0].items():
        if key == "audio":
            continue

        print(key, type(val))

    print(test_processed_dict[0].keys())
    print(test_processed_dict[0]["style_tags"])

    # # print(test_processed_dict['lyrics'])
    # if isinstance(test_processed_dict, dict):
    #     dct = {k: v for k, v in test_processed_dict.items() if k not in [
    #         "meta", "audio", "audio_dummy"
    #     ]}
    #     print(test_processed_dict['audio'].shape)
    # else:
    #     dct = test_processed_dict
    # write_json(dct, "tests/unittests/dataio/assets/processed.json")


def test_all_transform_with_batch_transform():

    cfg_path = "tests/unittests/dataio/assets/v5_dataloader_mariana.yaml"
    with open(cfg_path) as f:
        kwargs = yaml.full_load(f)
    print(kwargs)
    item_transform_list = kwargs["data"]["train_item_transform"]
    batch_transform_list = kwargs["data"]["train_batch_transform"]
    print(item_transform_list)

    train_item_transform = build_item_augmentation(item_transform_list)
    train_batch_transform = build_draw_batch_fn(batch_transform_list)

    test_init_dict = make_data()
    test_processed_list = train_item_transform(test_init_dict)

    test_batched_dict = train_batch_transform(test_processed_list)

    for key, value in test_batched_dict.items():
        print(key)

        if isinstance(value, torch.Tensor):
            print(value.shape)
        else:
            print(value)

        print("===========")
    print(test_batched_dict["target_tokens_length"])


def test_eval_item_transforms():
    cfg_path = "tests/unittests/dataio/assets/v5_dataloader_mariana.yaml"
    with open(cfg_path) as f:
        kwargs = yaml.full_load(f)

    item_transform_list = kwargs["data"]["predict_item_transform"]
    batch_transform_list = kwargs["data"]["predict_batch_transform"]

    train_item_transform = build_item_augmentation(item_transform_list)

    test_init_dict = make_eval_data()
    test_processed_list = train_item_transform(test_init_dict)

    print(test_processed_list.keys())

    train_batch_transform = build_draw_batch_fn(batch_transform_list)
    batch = train_batch_transform([test_processed_list])

    print(batch.keys())


def test_eval_item_transforms_cfg():
    cfg_path = "tests/unittests/dataio/assets/v5_dataloader_mariana_CFG.yaml"
    with open(cfg_path) as f:
        kwargs = yaml.full_load(f)

    item_transform_list = kwargs["data"]["predict_item_transform"]
    train_item_transform = build_item_augmentation(item_transform_list)

    test_init_dict = make_eval_data()
    test_processed_list = train_item_transform(test_init_dict)

    print(test_processed_list.keys())


def test_eval_item_transforms():
    cfg_path = "tests/unittests/dataio/assets/v5_dataloader_mariana.yaml"
    with open(cfg_path) as f:
        kwargs = yaml.full_load(f)

    item_transform_list = kwargs["data"]["predict_item_transform"]
    batch_transform_list = kwargs["data"]["predict_batch_transform"]

    train_item_transform = build_item_augmentation(item_transform_list)

    test_init_dict = make_eval_data()
    test_processed_list = train_item_transform(test_init_dict)

    print(test_processed_list.keys())

    train_batch_transform = build_draw_batch_fn(batch_transform_list)
    batch = train_batch_transform([test_processed_list])

    print(batch.keys())


def test_eval_item_transforms_cfg():
    cfg_path = "tests/unittests/dataio/assets/v5_dataloader_mariana_CFG.yaml"
    with open(cfg_path) as f:
        kwargs = yaml.full_load(f)

    item_transform_list = kwargs["data"]["predict_item_transform"]
    train_item_transform = build_item_augmentation(item_transform_list)

    test_init_dict = make_eval_data()
    test_processed_list = train_item_transform(test_init_dict)

    print(test_processed_list.keys())


def test_eval_item_transforms():
    cfg_path = "tests/unittests/dataio/assets/v5_dataloader_mariana.yaml"
    with open(cfg_path) as f:
        kwargs = yaml.full_load(f)

    item_transform_list = kwargs["data"]["predict_item_transform"]
    batch_transform_list = kwargs["data"]["predict_batch_transform"]

    train_item_transform = build_item_augmentation(item_transform_list)

    test_init_dict = make_eval_data()
    test_processed_list = train_item_transform(test_init_dict)

    print(test_processed_list.keys())

    train_batch_transform = build_draw_batch_fn(batch_transform_list)
    batch = train_batch_transform([test_processed_list])

    print(batch.keys())


def test_eval_item_transforms_cfg():
    cfg_path = "tests/unittests/dataio/assets/v5_dataloader_mariana_CFG.yaml"
    with open(cfg_path) as f:
        kwargs = yaml.full_load(f)

    item_transform_list = kwargs["data"]["predict_item_transform"]
    train_item_transform = build_item_augmentation(item_transform_list)

    test_init_dict = make_eval_data()
    test_processed_list = train_item_transform(test_init_dict)

    print(test_processed_list.keys())


def test_token_transform():
    test_init_dict = make_data_token()
    cfg_path = "tests/unittests/dataio/assets/v5_dataloader_mariana_offline_token.yaml"
    with open(cfg_path) as f:
        kwargs = yaml.full_load(f)

    item_transform_list = kwargs["data"]["train_item_transform"]
    train_item_transform = build_item_augmentation(item_transform_list)

    test_processed_list = train_item_transform(test_init_dict)
    if isinstance(test_processed_list, list):
        test_processed_list = test_processed_list[0]

    print(test_processed_list.keys())
    print(test_processed_list["target_token_ids"])


if __name__ == "__main__":
    # test_LyricsTransform()
    # test_all_transforms()
    # test_structure_parser()
    # test_style_tag_parser_and_freeform_parser()
    # test_all_transform_with_batch_transform()
    # test_eval_item_transforms()

    test_token_transform()
