import json
from webdataset import filters, shardlists
import numpy as np

import ffmpeg
import random
from recipes.mir2.parquet_dataset.vocab_index import get_genre_timbre_map 
from recipes.mir2.parquet_dataset.base import BaseParquetDataset

class MusicFmParquetDataset(BaseParquetDataset):
    def __init__(
        self,
        data_id,
        batch_size: int = 20,
        infer_batch_size: int = 32,
        resampled: bool =True,
        shuffle_buffer: int = 100,
        sample_rate: int = 24000,
        sample_len: float = 29.1,
        nodesplitter = shardlists.split_by_node,
        tag_types: list = ['genre', 'vocal_timbre'],
        validation: bool = False,
        test: bool = False,
        debug=False,
    ):
        super().__init__(
            data_id=data_id,
            batch_size=batch_size,
            infer_batch_size=infer_batch_size,
            resampled=resampled,
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
        # print(f' ******** sample keys: {sample.keys()}')
        # self.export_json(sample, meta=json.loads(sample['meta']))
        meta = json.loads(sample['meta']).get("raw", {})
        uutid = self.get_id(str(sample['uttid']))

        meta = json.loads(sample['meta'])
        if 'audio_tags' not in meta:
            raise Exception(f'audio_tags not in meta: {uutid}')
        
        music_tagging = meta['audio_tags']
        vec_dic = {}
        for key in self.tag_types:
            tags = music_tagging[key]
            vector = np.zeros(len(get_genre_timbre_map()[key]), dtype = np.float64)
            tag_map = get_genre_timbre_map()[key]
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                if tag not in tag_map:
                    return None
                oneshot = tag_map[tag]
                vector[oneshot] = 1
            vec_dic[key] = vector
        
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
    import webdataset as wds
    urls = "pipe:hdfs dfs -cat /home/byte_speech_sv/mulan/playlist_v5_uio/audio_shards/shards_{0000..2596}.tar"
    dataset = wds.WebDataset(urls=urls, resampled=True, shardshuffle=True, handler=wds.warn_and_continue)
    for data in dataset:
        print(data.keys())
        for key in data.keys():
            if key == 'audio.npy':
                print(key, data[key], data.shape)

def test2():
    import pickle
    dataset = MusicFmParquetDataset(data_id=3540, batch_size=20, validation=True)
    all_ids = set()
    for item in dataset:
        print(item[0].shape, [(k, v.shape) for k, v in item[1].items()], item[2].shape)
        pickle.dump(item[0], open('dump_audio.pkl', 'wb'))
        all_ids.update(item[2])
        # for key in data.keys():
        #     print(data[key])
    print(len(all_ids))
    #pickle.dump(np.concatenate(tag_scene), open('dump_tag.pkl', 'wb'))

if __name__ == '__main__':
    test2()
