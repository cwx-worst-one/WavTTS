import json
import webdataset as wds
from webdataset import filters, shardlists
import numpy as np

from scipy.signal import resample_poly
#from recipes.mir2.parquet_dataset.vocab_index import bigmusic_tags
from recipes.mir2.parquet_dataset.base import BaseParquetDataset
from recipes.mir2.parquet_dataset.new_label_process import GetTags, BinalizeTags, FilterByUttid
from recipes.mir2.parquet_dataset.vocab_index import BIGMUSIC_TAG_INFO, BIGMUSIC_TAG_INFO_GENRE_ONLY

class MusicFmParquetDataset(BaseParquetDataset):
    def __init__(
        self,
        data_id: int = 3309,
        batch_size: int = 12,
        infer_batch_size: int = 32,
        resampled: bool =True,
        shuffle_buffer: int = 100,
        sample_rate: int = 24000,
        sample_len: float = 29.1,
        nodesplitter = shardlists.split_by_node,
        tag_types = BIGMUSIC_TAG_INFO, #list = ['genre', 'mood', 'scene', 'vocal_gender', 'vocal_timbre'],
        validation: float = False,
        test: bool = False,
        num_iter: int = None,
        debug: bool = False,
        filter_uuid_list_hpath: str = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/rui.xia/bigmusic/mir_tasks/tagging_misc/uttid_7905.lst",
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
            num_iter=num_iter,
            debug=debug,
        )
        self.validation = validation
        self.test = test
        self.tag_types = tag_types
     
        self.new_FilterByUttid = FilterByUttid(
            # filter_list_file = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/rui.xia/bigmusic/mir_tasks/tagging_misc/uttid_4144.lst",
            # filter_list_file = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/rui.xia/bigmusic/mir_tasks/tagging_misc/uttid_7818.lst",
            # filter_list_file = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/rui.xia/bigmusic/mir_tasks/tagging_misc/uttid_7967.lst",
            # filter_list_file = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/rui.xia/bigmusic/mir_tasks/tagging_misc/uttid_7905.lst",
            filter_list_file = filter_uuid_list_hpath,
            key = "uttid"
        )

        self.new_GetTags = GetTags(
            tagging_params = self.tag_types,
            out_key = "tags",
            enable_filter = False if (validation or test) else True,
            skip_error = True if (validation or test) else False,
        )

        self.new_BinarizeTags = BinalizeTags(
            tagging_params = self.tag_types,
            in_key = "tags",
            out_key = "vec_dic"
        )

 
    def process_label_and_audio(self, sample):
        # meta = json.loads(sample['meta'])
        # uutid = self.get_id(str(sample['uttid']))
        # music_tagging = meta['audio_tags']
        # vec_dic = {}
        # for key in self.tag_types:
        #     tags = music_tagging[key]
        #     vector = np.zeros(len(bigmusic_tags()[key]), dtype = np.float64) 
        #     tag_map = bigmusic_tags()[key]
        #     if isinstance(tags, str):
        #         tags = [tags]
        #     for tag in tags:
        #         if tag in tag_map:
        #             oneshot = tag_map[tag]
        #             vector[oneshot] = 1
        #     vec_dic[key] = vector

        if not (self.validation or self.test):
            sample = self.new_FilterByUttid(sample)
        item = self.new_GetTags(sample)
        item = self.new_BinarizeTags(item)

        np_audio = sample["wav"]
        src = sample['src_sample_rate']
        np_audio = np.frombuffer(np_audio, dtype=np.int16)
        np_audio =  np_audio / 32768.0
        if src != self.sample_rate:
            np_audio = resample_poly(np_audio, self.sample_rate, src)
        np_audio = np_audio[np.newaxis, :]
        
        item = {
            # uuid_int is needed because we need to all_gather the data to deduplicate the data
            # in both training, validation, and test
            
            'uuid_int': self.get_id(str(item['uttid'])),
            'uuid': str(item['uttid']),
            'audio.npy': np_audio,
            'tags': item['vec_dic'],
            'index_url': item['__index_url__'],
        }
        return item


def test2():
    dataset = MusicFmParquetDataset(data_id=[4144], resampled=False, batch_size=32, validation=True,
    nodesplitter=shardlists.single_node_only)
    all_ids = set()
    for item in dataset:
        print(item[0].shape, [(k, v.shape) for k, v in item[1].items()], item[2].shape)
        #from IPython import embed; embed(using=False); 
        all_ids.update(item[2])
        # for key in data.keys():
        #     print(data[key])
    print(len(all_ids))
    #pickle.dump(np.concatenate(tag_scene), open('dump_scene.pkl', 'wb'))
            
def test3():
    import pickle
    dataset = MusicFmParquetDataset(data_id=4144, batch_size=1, infer_batch_size=32, test=True)
    all_ids = set()
    for item in dataset:
        print(item[0].shape, [(k, v.shape) for k, v in item[1].items()], item[2].shape)
        all_ids.update(item[2])
        # for key in data.keys():
        #     print(data[key])

def test_5004():
    from tqdm import tqdm
    dataset = MusicFmParquetDataset(
        data_id=5004,
        batch_size=8,
        infer_batch_size=32,
        shuffle_buffer=100,
        test=False,
        resampled=True,
        debug=False,
        tag_types=BIGMUSIC_TAG_INFO_GENRE_ONLY,
    )
    count = 0
    for item in tqdm(dataset):
        count += 1
        # if count > 1000:
        #     break

def test(data_id=4144):
    from tqdm import tqdm
    dataset = MusicFmParquetDataset(
        data_id=data_id,
        batch_size=8,
        infer_batch_size=32,
        shuffle_buffer=0,
        test=True,
        resampled=False,
        debug=False,
        tag_types=BIGMUSIC_TAG_INFO,
        # tag_types=BIGMUSIC_TAG_INFO_GENRE_ONLY,
    )
    count = 0
    for item in tqdm(dataset):
        count += 1
        if count > 5:
            break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add a data_id argument')
    parser.add_argument('--data_id', type=str, help='The data ID to process')
    args = parser.parse_args()
    data_id = int(args.data_id) if args.data_id else 4144
    if args.data_id:
        print(f"Processing data with ID: {args.data_id}")
        test(data_id=data_id)
    else:
        print("No data ID provided.")
        test(data_id=4144)
