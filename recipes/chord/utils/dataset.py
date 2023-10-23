import braceexpand
import webdataset
import functools
from torch.utils.data import DataLoader
import os

from samantha.utils.webdataset import apply_webdataset_pipeline
from recipes.chord.preprocess.common import ChordPreprocessor
from recipes.beat.utils.dataset import MCC_dataset, GalaxyArk_dataset, hot_galaxy_dataset, GalaxyArk_new_dataset


BYTENAS_NAME = os.listdir('/mnt/bn/')[0]


def get_train_pipeline(preprocessor, shuffle_buffer, batch_size, num_iter, to_tuple):
    pipeline = []
    
    #pipeline.append(functools.partial(webdataset.WebDataset.shuffle, shuffle_buffer))
    pipeline.append(functools.partial(webdataset.WebDataset.decode, handler=webdataset.warn_and_stop))

    proprocess = functools.partial(ChordPreprocessor.train_batch_preprocess, preprocessor)
    pipeline.append(functools.partial(webdataset.WebDataset.compose, proprocess))
    pipeline.append(functools.partial(webdataset.WebDataset.shuffle, shuffle_buffer))
    pipeline.append(functools.partial(webdataset.WebDataset.to_tuple, to_tuple))
    pipeline.append(functools.partial(webdataset.WebDataset.batched, batch_size))
    pipeline.append(functools.partial(webdataset.WebDataset.with_epoch, num_iter))
    return pipeline


def get_val_pipeline(preprocessor):
    pipeline = []    
    pipeline.append(functools.partial(webdataset.WebDataset.decode))
    proprocess = functools.partial(ChordPreprocessor.val_preprocess, preprocessor)
    pipeline.append(functools.partial(webdataset.WebDataset.map, proprocess))
    pipeline.append(functools.partial(webdataset.WebDataset.to_tuple, "audio chord_root chord_triad chord_note chord_ignore __key__"))
    pipeline.append(functools.partial(webdataset.WebDataset.batched, 1))
    return pipeline


def supervised_dataset():
    supervised_urls = []
    urls = [
        "/mnt/bd/sheetdoctor/chord/billboard_chord/train/shards-{0000..0016}.tar",
        "/mnt/bd/sheetdoctor/chord/isophonic_chord/train/shards-0000.tar",
        "/mnt/bd/sheetdoctor/chord/leadsheet_chord/train/shards-{0000..0022}.tar",
        "/mnt/bd/sheetdoctor/chord/rwc_chord/train/shards-{0000..0002}.tar",
        "/mnt/bd/sheetdoctor/chord/uspop_chord/train/shards-{0000..0005}.tar",
    ]
    for url in urls:
        supervised_urls = supervised_urls + list(braceexpand.braceexpand(url))
    
    return supervised_urls


def resso_dataset():
    semi_supervised_urls = []
    resso_urls = open("recipes/chord/conf/resso_mss_shard_list.v4.txt", "r")
    for url in resso_urls.readlines():
        semi_supervised_urls.append(f"pipe: hdfs dfs -cat {url.strip()}" )

    return semi_supervised_urls


def mingus_dataset():
    semi_supervised_urls = []
    resso_urls = open("recipes/chord/conf/mingus_list.txt", "r")
    for url in resso_urls.readlines():
        semi_supervised_urls.append(f"pipe: hdfs dfs -cat {url.strip()}" )

    return semi_supervised_urls


def karaoke_dataset():
    url = list(braceexpand.braceexpand("/mnt/bd/sheetdoctor/beat/karaoke_beat/train/shards-{0000..0043}.tar"))
    return url


def jaychou_dataset():
    url = ["pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/chord/jaychou_chord/validation/*.tar"]
    return url


def pop909_dataset():
    url = list(braceexpand.braceexpand("pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/chord/pop909_chord/validation/shards-{0000..0004}.tar"))
    return url


def leadsheet_eval_dataset():
    url = list(braceexpand.braceexpand("/mnt/bd/sheetdoctor/chord/leadsheet_chord/validation/shards-{0000..0005}.tar"))
    return url


def get_train_dataset(dataset_names, preprocessor, shuffle_buffer, batch_size, num_iter, num_workers, pin_memory, to_tuple):
    dataset_urls = []
    for dataset_name in dataset_names:
        if dataset_name == "karaoke":
            dataset_urls += karaoke_dataset()
        elif dataset_name == "resso":
            dataset_urls += resso_dataset()
        elif dataset_name == "labeled":
            dataset_urls += supervised_dataset()
        elif dataset_name == 'mcc':
            dataset_urls += MCC_dataset()
        elif dataset_name == 'mingus':
            dataset_urls += mingus_dataset()
        elif dataset_name == 'hot_galaxy':
            dataset_urls += hot_galaxy_dataset()
        elif dataset_name == 'galaxyark':
            dataset_urls += GalaxyArk_dataset()
        elif dataset_name == 'new_galaxyark':
            dataset_urls += GalaxyArk_new_dataset()
        else:
            raise "Only support these datasets: [karaoke, resso, labeled, hot_galaxy, galaxyark, new_galaxyark]"

    dataset = webdataset.WebDataset(urls=dataset_urls, resampled=True)
    dataset = dataset.with_length(num_iter*num_workers)
    train_transformed = apply_webdataset_pipeline(wds_dataset=dataset, pipeline=get_train_pipeline(preprocessor, shuffle_buffer, batch_size, num_iter, to_tuple))
    train_dataloader = DataLoader(dataset=train_transformed, shuffle=False, batch_size=None, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    
    return train_dataloader


def split_by_node(src):
    for s in src:
        yield s


def get_validation_dataset(dataset_names, preprocessor, num_workers, pin_memory):
    dataset_urls = []
    for dataset_name in dataset_names:
        if dataset_name == "leadsheet":
            dataset_urls += leadsheet_eval_dataset()
        elif dataset_name == "jaychou":
            dataset_urls += jaychou_dataset()
        elif dataset_name == 'pop909':
            dataset_urls += pop909_dataset()
        else:
            raise "Only support these datasets: [jaychou, leadsheet, pop909]"

    dataset = webdataset.WebDataset(urls=dataset_urls, nodesplitter=split_by_node)
    validate_transformed = apply_webdataset_pipeline(wds_dataset=dataset, pipeline=get_val_pipeline(preprocessor))
    validate_dataloader = DataLoader(dataset=validate_transformed, shuffle=False, batch_size=None, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    return validate_dataloader
