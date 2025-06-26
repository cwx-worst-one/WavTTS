import json
from webdataset import filters, shardlists
import numpy as np

from scipy.signal import resample_poly
from recipes.mir2.parquet_dataset.vocab_index import (
    KARAOKE_TAGS, tempo_convert, year_convert, karaoke_key_convert)
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
        tag_types: list = ['genres', 'key', 'tempo', 'year', 'instruments'],
        validation: bool = False,
        test: bool = False,
        debug: bool = False,
        replacement: bool = False,
        num_iter: int = None,
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
            replacement=replacement,
            num_iter=num_iter,
        )
        self.tag_types = tag_types
       
    def process_label_and_audio(self, sample):
        print(f' ******** sample keys: {sample.keys()}')
        # self.export_json(sample, meta=json.loads(sample['meta']))
        meta = json.loads(sample['meta']).get("raw", {})
        lyrics = json.loads(sample['meta']).get("lyrics", {})
        uutid = self.get_id(str(sample['uttid']))

        vec_dic = {}
        for tag_type in self.tag_types:
            raw_tags = str(meta[tag_type])
            tags = raw_tags.replace('\'', '').split(',')
            vector = np.zeros(len(KARAOKE_TAGS[tag_type]), dtype = np.float64)
            tag_map = KARAOKE_TAGS[tag_type]
            tag_map = {tt.strip().lower(): idx for tt, idx in tag_map.items() }

            if tag_type == 'tempo':
                try:        
                    bpm = int(round(float(tags[0])))      
                    for tag, bpm_range in tempo_convert.items():
                        if bpm >= bpm_range[0] and bpm <= bpm_range[1]:
                            vector[tag_map[tag]] = 1 
                            #print(f'***** {meta["song_id"]} {tag_type} {tag} {bpm}')                  
                except:
                    vector[tag_map["none"]] = 1
            elif tag_type == 'year':       
                year = int(tags[0])    
                for tag, year_range in year_convert.items():
                    if year >= year_range[0] and year <= year_range[1]:
                        vector[tag_map[tag]] = 1 
                        #print(f'***** {meta["song_id"]} {tag_type} {tag} {year}')                  
            else:
                for tag in tags:
                    tag = tag.replace('\"', '').strip().lower()
                    if tag_type == 'genres':
                        tag = 'pop' if tag == 'french pop music' else tag
                    
                    if tag_type == 'key':
                        tag = karaoke_key_convert[tag]
                    #print(f'***** {meta["song_id"]} {tag_type} {tag}')
                    if tag in tag_map:
                        vector[tag_map[tag]] = 1
            
            vec_dic[tag_type] = vector
        
        np_audio = sample["wav"]
        src = sample['src_sample_rate']
        np_audio = np.frombuffer(np_audio, dtype=np.int16)
        np_audio =  np_audio / 32768.0
        if src != self.sample_rate:
            np_audio = resample_poly(np_audio, self.sample_rate, src)
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
    dataset = MusicFmParquetDataset(data_id=6667, batch_size=20, validation=False, debug=True)
    all_ids = set()
    for item in dataset:
        #from IPython import embed; embed(using=False)
        print(item[0].shape, [(k, v.shape) for k, v in item[1].items()], item[2].shape)
        #pickle.dump(item[0], open('dump_audio.pkl', 'wb'))
        all_ids.update(item[2])
        # for key in data.keys():
        #     print(data[key])
    print(len(all_ids))
    #pickle.dump(np.concatenate(tag_scene), open('dump_tag.pkl', 'wb'))

if __name__ == '__main__':
    test2()
