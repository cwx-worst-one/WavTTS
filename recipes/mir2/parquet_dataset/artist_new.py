import json
from webdataset import filters, shardlists
import numpy as np

from scipy.signal import resample_poly
from recipes.mir2.parquet_dataset.vocab_index import artist_new, ARTIST_NAME_LIST
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
        tag_types = ['artist'],
        validation = False,
        test = False,
        debug = False,
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
        #print(f' ******** sample keys: {sample.keys()}')
        # self.export_json(sample, meta=json.loads(sample['meta']))
        
        meta = json.loads(sample['meta'])
        uutid = self.get_id(str(sample['uttid']))
        # if 'raw' in meta:
        #     print(meta['raw'])

        vec_dic = {}
        for tag_type in self.tag_types:
            artist_name = []
            artist = meta.get('raw', {}).get('artist', '').replace('"', '')
            if artist and artist in ARTIST_NAME_LIST:
                artist_name.append(artist)
            artist = meta.get('raw', {}).get('artists', '')
            if artist and isinstance(artist, list):
                for artist_ in artist:
                    artist_ = artist_.replace('"', '')
                    if artist_ in ARTIST_NAME_LIST:
                        artist_name.append(artist_)          
            artist = meta.get('artist_name', '')
            if artist and artist in ARTIST_NAME_LIST:
                artist_name.append(artist)
            artist = meta.get('raw', {}).get('song_detail', {}).get('ar', [{}])[0].get('name', '')
            if artist and artist in ARTIST_NAME_LIST:
                artist_name.append(artist)
            artist = meta.get('artist', '')
            if artist and artist in ARTIST_NAME_LIST:
                artist_name.append(artist)
            artist = meta.get('meta_song_author_name', '')
            if artist and artist in ARTIST_NAME_LIST:
                artist_name.append(artist)
            artist = meta.get('raw', {}).get('main_artist', '')
            if artist and artist in ARTIST_NAME_LIST:
                artist_name.append(artist)
            artist = meta.get('raw', {}).get('歌手名', '')
            if artist and artist in ARTIST_NAME_LIST:
                artist_name.append(artist)
            tags = list(set(artist_name))

            vector = np.zeros(len(artist_new()[tag_type]), dtype = np.float64) 
            tag_map = artist_new()[tag_type]

            for tag in tags:
                if tag in tag_map:
                    vector[tag_map[tag]] = 1
                    #print("found", tag)
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

def test2():
    import pickle
    dataset = MusicFmParquetDataset(data_id=[5446, 3615, 3571], batch_size=20, validation=True, debug=True)
    all_ids = set()
    for item in dataset:
        print(item[0].shape, [(k, v.shape) for k, v in item[1].items()], item[2].shape)
        all_ids.update(item[2])
        # for key in data.keys():
        #     print(data[key])
    print(len(all_ids))
    #pickle.dump(np.concatenate(tag_scene), open('dump_scene.pkl', 'wb'))

if __name__ == '__main__':
    test2()
