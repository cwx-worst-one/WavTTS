import braceexpand
import webdataset
import functools
from torch.utils.data import DataLoader
import os

from samantha.utils.webdataset import apply_webdataset_pipeline
from recipes.beat.preprocess.common import BeatPreprocessor

def get_train_pipeline(preprocessor, shuffle_buffer, batch_size, num_iter, to_tuple):
    pipeline = []
    
    #pipeline.append(functools.partial(webdataset.WebDataset.shuffle, shuffle_buffer))
    pipeline.append(functools.partial(webdataset.WebDataset.decode, handler=webdataset.warn_and_stop))

    proprocess = functools.partial(BeatPreprocessor.train_batch_preprocess, preprocessor)
    pipeline.append(functools.partial(webdataset.WebDataset.compose, proprocess))
    pipeline.append(functools.partial(webdataset.WebDataset.shuffle, shuffle_buffer))
    pipeline.append(functools.partial(webdataset.WebDataset.to_tuple, to_tuple))
    pipeline.append(functools.partial(webdataset.WebDataset.batched, batch_size))
    pipeline.append(functools.partial(webdataset.WebDataset.with_epoch, num_iter))
    return pipeline


def get_val_pipeline(preprocessor):
    pipeline = []    
    pipeline.append(functools.partial(webdataset.WebDataset.decode))
    proprocess = functools.partial(BeatPreprocessor.val_preprocess, preprocessor)
    pipeline.append(functools.partial(webdataset.WebDataset.map, proprocess))
    pipeline.append(functools.partial(webdataset.WebDataset.to_tuple, "audio.npy beats.pickle tempo_label orig_beats dataset.txt __key__"))
    pipeline.append(functools.partial(webdataset.WebDataset.batched, 1))
    return pipeline


def _beat_supervised_dataset():
    supervised_urls = []
    urls = [
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/ballroom_beat/train/shards-0000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/beatles_beat/train/shards-0000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/hainsworth_beat/train/shards-0000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/hjdb_beat/train/shards-0000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/rwc_beat/train/shards-{0000..0001}.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/simac_beat/train/shards-0000.tar",
        "pipe: hdfs dfs -cat \
            hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/harmonix_beat/train/shards-{0000..0004}.tar",
        "pipe: hdfs dfs -cat \
            hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/smc_beat/train/shards-0000.tar",
        "pipe: hdfs dfs -cat \
            hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/karaoke_beat/train/shards-{0000..0043}.tar"
    ]
    for url in urls:
        supervised_urls = supervised_urls + list(braceexpand.braceexpand(url))
    
    return supervised_urls


def beat_supervised_dataset():
    supervised_urls = []
    urls = [
        "/mnt/bn/mir-tasks/beat/ballroom_beat/train/shards-0000.tar",
        "/mnt/bn/mir-tasks/beat/beatles_beat/train/shards-0000.tar",
        "/mnt/bn/mir-tasks/beat/hainsworth_beat/train/shards-0000.tar",
        "/mnt/bn/mir-tasks/beat/hjdb_beat/train/shards-0000.tar",
        "/mnt/bn/mir-tasks/beat/rwc_beat/train/shards-{0000..0001}.tar",
        "/mnt/bn/mir-tasks/beat/simac_beat/train/shards-0000.tar",
        "/mnt/bn/mir-tasks/beat/harmonix_beat/train/shards-{0000..0004}.tar",
        "/mnt/bn/mir-tasks/beat/smc_beat/train/shards-0000.tar",
        "/mnt/bn/mir-tasks/beat/karaoke_beat/train/shards-{0000..0043}.tar"
    ]
    for url in urls:
        supervised_urls = supervised_urls + list(braceexpand.braceexpand(url))
    
    return supervised_urls


def vocalbeat_supervised_dataset():
    karaoke_urls = list(braceexpand.braceexpand("pipe: hdfs dfs -cat \
         hdfs://harunava/home/byte_speech_sv/data/webdataset/vocal_beat/karaoke_vocal_beat/train/shards-{0000..0043}.tar"))
    karaoke_stretch_urls = list(braceexpand.braceexpand("pipe: hdfs dfs -cat \
         hdfs://harunava/home/byte_speech_sv/data/webdataset/vocal_beat/karaoke_vocal_stretch_beat/train/shards-{0000..0046}.tar"))
    return karaoke_urls + karaoke_stretch_urls


def MCC_dataset():
    semi_supervised_urls = []
    resso_urls = open("recipes/beat/conf/license_based_mcc.label_C.v1.txt", "r")
    for url in resso_urls.readlines():
        semi_supervised_urls.append(f"pipe: hdfs dfs -cat {url.strip()}" )

    return semi_supervised_urls


def MCC_MSS_dataset():
    semi_supervised_urls = []
    resso_urls = open("recipes/beat/conf/license_based_mcc_mss.label_C.v1.txt", "r")
    for url in resso_urls.readlines():
        semi_supervised_urls.append(f"pipe: hdfs dfs -cat {url.strip()}" )

    return semi_supervised_urls


def _hot_galaxy_dataset():
    urls = [
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/00000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/00500.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/01000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/01500.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/02000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/02500.tar",
    ]
    return urls


def hot_galaxy_dataset():
    urls = [
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_3k/shards-0000.tar",
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_3k/shards-0001.tar",
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_3k/shards-0002.tar",
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_3k/shards-0003.tar",
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_3k/shards-0004.tar",
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_3k/shards-0005.tar",
    ]
    return urls


def _GalaxyArk_dataset():
    semi_supervised_urls = []
    resso_urls = open("recipes/beat/conf/GalaxyArk_tar_list.txt", "r")
    for url in resso_urls.readlines():
        semi_supervised_urls.append(f"pipe: hdfs dfs -cat {url.strip()}" )

    return semi_supervised_urls


def GalaxyArk_dataset():
    semi_supervised_urls = []
    for file in os.listdir('/mnt/bn/mir-tasks/hot_galaxy/galaxyark'):
        semi_supervised_urls.append(f"/mnt/bn/mir-tasks/hot_galaxy/galaxyark/{file}")

    return semi_supervised_urls


def GalaxyArk_new_dataset():
    semi_supervised_urls = []
    for file in os.listdir('/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark'):
        semi_supervised_urls.append(f"/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark/{file}")

    return semi_supervised_urls


def GalaxyArk_new_dataset_mss():
    semi_supervised_urls = []
    for file in os.listdir('/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_mss'):
        semi_supervised_urls.append(f"/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_mss/{file}")

    return semi_supervised_urls


def GalaxyArk_new_vocal_dataset():
    semi_supervised_urls = []
    for file in os.listdir('/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_vocal'):
        semi_supervised_urls.append(f"/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_vocal/{file}")

    return semi_supervised_urls


def _hot_galaxy_vocal_dataset():
    urls = [
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mss_shard/00000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mss_shard/00500.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mss_shard/01000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mss_shard/01500.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mss_shard/02000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mss_shard/02500.tar",
    ]
    return urls


def hot_galaxy_vocal_dataset():
    urls = [
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_vocal/shards-0000.tar",
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_vocal/shards-0001.tar",
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_vocal/shards-0002.tar",
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_vocal/shards-0003.tar",
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_vocal/shards-0004.tar",
        "/mnt/bn/mir-tasks/hot_galaxy/galaxyark_vocal/shards-0005.tar",
    ]
    return urls

def resso_dataset():
    semi_supervised_urls = []
    resso_urls = open("recipes/chord/conf/resso_mss_shard_list.v4.txt", "r")
    for url in resso_urls.readlines():
        semi_supervised_urls.append(f"pipe: hdfs dfs -cat {url.strip()}" )

    return semi_supervised_urls


def bytebeat_dataset():
    url = ["pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/bytebeat_beat/validation/*.tar"]
    return url


def clip500_dataset():
    url = ["pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/clip500_beat/test/*.tar"]
    return url


def gtzan_dataset():
    url = ["pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/beat/gtzan_beat/validation/*.tar"]
    return url


def MCC126_dataset():
    url = ["pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/webdataset/vocal_beat/MCC126_vocal_beat/validation/shards-0000.tar"]
    return url


def MCC32_dataset():
    url = ["/mnt/bn/mir-tasks/beat_vocal/MCC32_vocal_beat/test/shards-0000.tar"]
    return url


def get_train_dataset(dataset_names, preprocessor, shuffle_buffer, batch_size, num_iter, num_workers, pin_memory, to_tuple):
    dataset_urls = []
    for dataset_name in dataset_names:
        if dataset_name == "mcc":
            dataset_urls += MCC_dataset()
        elif dataset_name == 'mcc_mss':
            dataset_urls += MCC_MSS_dataset()
        elif dataset_name == "resso":
            dataset_urls += resso_dataset()
        elif dataset_name == "beat_labeled":
            dataset_urls += beat_supervised_dataset()
        elif dataset_name == 'vocalbeat_labeled':
            dataset_urls += vocalbeat_supervised_dataset()
        elif dataset_name == 'hot_galaxy':
            dataset_urls += hot_galaxy_dataset()
        elif dataset_name == "hot_galaxy_vocal":
            dataset_urls += hot_galaxy_vocal_dataset()
        elif dataset_name == 'galaxyark':
            dataset_urls += GalaxyArk_dataset()
        elif dataset_name == 'new_galaxyark':
            dataset_urls += GalaxyArk_new_dataset()
        elif dataset_name == 'new_galaxyark_mss':
            dataset_urls += GalaxyArk_new_dataset_mss()
        else:
            raise "Only support these datasets: [mcc, mcc_mss, resso, beat_labeled, hot_galaxy, hot_galaxy_vocal, galaxyark, new_galaxyark, new_galaxyark_mss]"

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
        if dataset_name == "clip500":
            dataset_urls += clip500_dataset()
        elif dataset_name == "bytebeat":
            dataset_urls += bytebeat_dataset()
        elif dataset_name == 'gtzan':
            dataset_urls += gtzan_dataset()
        elif dataset_name == 'mcc126':
            dataset_urls += MCC126_dataset()
        elif dataset_name == 'mcc32':
            dataset_urls += MCC32_dataset()
        else:
            raise "Only support these datasets: [clip500, bytebeat, gtzan, mcc126, mcc32]"

    dataset = webdataset.WebDataset(urls=dataset_urls, nodesplitter=split_by_node)
    validate_transformed = apply_webdataset_pipeline(wds_dataset=dataset, pipeline=get_val_pipeline(preprocessor))
    validate_dataloader = DataLoader(dataset=validate_transformed, shuffle=False, batch_size=None, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    return validate_dataloader