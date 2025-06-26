import json
import webdataset as wds
from webdataset import shardlists
import numpy as np
import random
import ffmpeg

from recipes.mir2.parquet_dataset.vocab_index import quality_tags
from recipes.mir2.parquet_dataset.base import BaseParquetDataset


class MusicFmParquetDataset(BaseParquetDataset):
    def __init__(
        self,
        data_id,
        batch_size: int = 12,
        infer_batch_size: int = 32,
        resampled: bool =True,
        num_iter: int = None,
        shuffle_buffer: int = 100,
        sample_rate: int = 24000,
        sample_len: float = 29.1,
        nodesplitter = shardlists.split_by_node,
        tag_types = ['quality'],
        validation: float = False,
        test: bool = False,
        debug=False,
    ):
        super().__init__(
            data_id=data_id,
            batch_size=batch_size,
            infer_batch_size=infer_batch_size,
            resampled=resampled,
            num_iter=num_iter,
            shuffle_buffer=shuffle_buffer,
            sample_rate=sample_rate,
            sample_len=sample_len,
            nodesplitter=nodesplitter,
            validation=validation,
            test=test,
            debug=debug,
        )
        self.tag_types = tag_types


    def process_label_and_audio(self, sample):
        #print(f' ******** sample keys: {sample.keys()}')
        # self.export_json(sample, meta=json.loads(sample['meta']))
        meta = json.loads(sample['meta'])
        tags = meta["filter_label"]
        uutid = self.get_id(str(sample['uttid']))

        vec_dic = {}
        for tag_type in self.tag_types:
            vector = np.zeros(len(quality_tags()[tag_type]), dtype = np.float64)

            if tag_type == 'quality':
                try:
                    if tags["high_quality"] == "yes":
                        vector[0] = 1
                    if tags["popular_potential"] == "yes":
                        vector[1] = 1
                    if tags["high_quality"] == "yes" and tags["popular_potential"] == "yes":
                        vector[2] = 1
                    if tags["high_quality"] == "no" and tags["popular_potential"] == "no":
                        vector[3] = 1
                except:   
                    raise Exception(f"can't find tags 'high_quality' or 'popular_potential'")
            
            vec_dic[tag_type] = vector

        subp = (
            ffmpeg.input("pipe:")
            .output(
                "pipe:1",  # set the the output to stdout
                format="wav",
                acodec="pcm_s16le",
                ac=1,
                ar=self.sample_rate,
            )
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        np_audio, errmsg = subp.communicate(sample["wav"])
        if subp.returncode != 0:
            raise Exception(f"failed to decode the audio: {errmsg.decode(errors='ignore')}")

        np_audio = np.frombuffer(np_audio, dtype=np.int16)
        np_audio =  np_audio / 32768.0
        np_audio = np_audio[np.newaxis, :]

        item = {
            'uuid': uutid,
            'audio.npy': np_audio,
            'tags': vec_dic,
        }
        return item

def test():
    urls = "pipe:hdfs dfs -cat /home/byte_speech_sv/mulan/playlist_v5_uio/audio_shards/shards_{0000..2596}.tar"
    dataset = wds.WebDataset(urls=urls, resampled=True, shardshuffle=True, handler=wds.warn_and_continue)
    for data in dataset:
        print(data.keys())
        for key in data.keys():
            if key == 'audio.npy':
                print(key, data[key], data.shape)
def test2():
    dataset = MusicFmParquetDataset(data_id=4338, batch_size=1, test=True, resampled=False, debug=True)
    all_ids = set()
    for item in dataset:
        print(item[0].shape, [(k, v.shape) for k, v in item[1].items()], item[2].shape)
        all_ids.update(item[2])
        # for key in data.keys():
        #     print(data[key])
    print(len(all_ids))

if __name__ == '__main__':
    test2()
