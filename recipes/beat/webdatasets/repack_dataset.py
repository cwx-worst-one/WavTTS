"""
Beatles Beat Tracking Dataset.
"""

import os
import random

import webdataset
import tqdm
import numpy as np
import soundfile as sf

from samantha.dataio.webdataset.builder import AbstractWebDatasetGeneratorBasedBuilder
from recipes.beat.utils.ffmpeg_utils import convert_audio_ffmpeg_bytes_to_bytes, convert_audio_ffmpeg_path_to_bytes


from samantha.dataio.webdataset.writer import ShardWriter


def rewrite_galaxyark_vocal():
    with ShardWriter(
        f"{'/mnt/bn/mir-tasks/hot_galaxy/galaxyark_vocal'}/shards-%04d.tar",
        maxcount=500,
        maxsize=10e9,
    ) as sink:
        for file in os.listdir('/mnt/bn/mir-tasks/hot_galaxy/mss_shard/'):
            datasets = webdataset.WebDataset('/mnt/bn/mir-tasks/hot_galaxy/mss_shard/'+file).decode()
            for data in tqdm.tqdm(datasets):
                np_acc_audio = convert_audio_ffmpeg_bytes_to_bytes(data["mss_acc"], 16000, True)
                np_vocal_audio = convert_audio_ffmpeg_bytes_to_bytes(data["mss_vocal"], 16000, True)
                example = {}
                example["dataset.txt"] = 'galaxyark_vocal'
                example["__key__"] = data['__key__']
                example["mss_acc"] = np_acc_audio
                example["mss_vocal"] = np_vocal_audio
                sink.write(example)

def rewrite_galaxyark():
    with ShardWriter(
        f"{'/mnt/bn/mir-tasks/hot_galaxy/galaxyark'}/shards-%04d.tar",
        maxcount=500,
        maxsize=10e9,
    ) as sink:
        file_list = open("recipes/beat/conf/GalaxyArk_tar_list.txt", "r")
        for file in file_list.readlines():
            datasets = webdataset.WebDataset(f"pipe: hdfs dfs -cat {file.strip()}")
            for data in tqdm.tqdm(datasets):
                np_audio = convert_audio_ffmpeg_bytes_to_bytes(data["mp3"], 16000, True)
                example = {}
                example["dataset.txt"] = 'galaxyark'
                example["__key__"] = data['__key__']
                example["mp3"] = np_audio
                sink.write(example)

def rewrite_galaxyark_3k():
    urls = [
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/00000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/00500.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/01000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/01500.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/02000.tar",
        "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/mp3_bytes_shard/02500.tar",
    ]

    with ShardWriter(
        f"{'/mnt/bn/mir-tasks/hot_galaxy/galaxyark_3k'}/shards-%04d.tar",
        maxcount=500,
        maxsize=10e9,
    ) as sink:
        for file in urls:
            datasets = webdataset.WebDataset(file).decode()
            for data in tqdm.tqdm(datasets):
                np_audio = convert_audio_ffmpeg_bytes_to_bytes(data["mp3"], 16000, True)
                example = {}
                example["dataset.txt"] = 'galaxyark_3k'
                example["__key__"] = data['__key__']
                example["mp3"] = np_audio
                sink.write(example)

#rewrite_galaxyark_3k()


def rewrite_new_galaxyark():
    with ShardWriter(
        f"{'/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_vocal'}/shards-%04d.tar",
        maxcount=500,
        maxsize=10e9,
    ) as sink:
        for file in os.listdir('/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_mp3_shard/'):
            if 'vocal' not in file:
                continue
            datasets = webdataset.WebDataset('/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_mp3_shard/'+file).decode()
            for data in tqdm.tqdm(datasets):
                np_audio = convert_audio_ffmpeg_bytes_to_bytes(data["mp3"], 16000, True)
                example = {}
                example["dataset.txt"] = 'galaxyark'
                example["__key__"] = data['__key__']
                example["mp3"] = np_audio
                sink.write(example)

#rewrite_new_galaxyark()


def write_audio():
    import soundfile as sf
    import io

    for file in os.listdir('/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_vocal/'):
        datasets = webdataset.WebDataset('/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_vocal/'+file).decode()
        for data in tqdm.tqdm(datasets):
            np_audio = convert_audio_ffmpeg_bytes_to_bytes(data["mp3"], 16000, True)
            np_audio, sr = sf.read(io.BytesIO(np_audio))
            sf.write('/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_audio/'+data['__key__']+'.wav', np_audio, 16000)
#write_audio()


def rewrite_new_galaxyark_mss():
    with ShardWriter(
        f"{'/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_mss'}/shards-%04d.tar",
        maxcount=500,
        maxsize=10e9,
    ) as sink:
        for file in os.listdir('/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_mss_shard/'):
            datasets = webdataset.WebDataset('/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_mss_shard/'+file).decode()
            for data in tqdm.tqdm(datasets):
                np_vocal_audio = convert_audio_ffmpeg_bytes_to_bytes(data["mss_vocal"], 16000, True)
                np_mix_audio = convert_audio_ffmpeg_path_to_bytes('/mnt/bn/mir-tasks/hot_galaxy/new_galaxyark_audio/'+data['__key__']+'.wav', 16000, True)
                example = {}
                example["dataset.txt"] = 'galaxyark_mss'
                example["__key__"] = data['__key__']
                example["mss_vocal"] = np_vocal_audio
                example["mss_mix"] = np_mix_audio
                sink.write(example)

rewrite_new_galaxyark_mss()