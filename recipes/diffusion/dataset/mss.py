import hashlib
import random
import logging
import os

import numpy as np
import torch
import webdataset as wds
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from webdataset.pipeline import DataPipeline

MSS_18_TO_9_STEM = {
    'vocal': 'vocals', 
    'bass': 'bass', 
    'drums': 'drums', 'percussive': 'drums',
    'guitar': 'guitar', 
    'piano': 'piano',
    'strings': 'strings',
    'reed': 'wind', 'brass': 'wind', 'pipe': 'wind',
    'synth lead': 'synth', 'synth pad': 'synth', 'chromatic percussion': 'synth', 'organ': 'synth',  'ensemble': 'synth', 'synth effects': 'synth',
    'other': 'other', 'sound effects': 'other' 
    }
MSS_18_TO_4_STEM = {
    'vocal': 'vocals', 
    'bass': 'bass', 
    'drums': 'drums', 'percussive': 'drums',
    'guitar': 'other', 'piano': 'other', 'strings': 'other', 'reed': 'other', 'brass': 'other', 'pipe': 'other', 'wind': 'other', 
    'synth': 'other', 'synth lead': 'other', 'synth pad': 'other', 'chromatic percussion': 'other', 'organ': 'other', 'ensemble': 'other', 
    'synth effects': 'other', 'other': 'other', 'sound effects': 'other'
    }

MSS_18_STEM = ['vocals', 'bass', 'drums', 'percussive', 'guitar', 'piano', 'strings', 
               'reed', 'brass', 'pipe', 'organ', 'synth lead', 'synth pad', 'synth effects', 
               'chromatic percussion', 'ensemble', 'other', 'sound effects']
MSS_9_STEM = ['vocals', 'bass', 'drums', 'guitar', 'piano', 'strings', 'wind', 'synth', 'other']
MSS_4_STEM = ['vocals', 'bass', 'drums', 'other']


def add_urls(urls, url_short):
    try:
        url_path, filename = os.path.split(url_short)
        filename, ext = os.path.splitext(filename)
        filename = filename.replace('{','').replace('}','')
        start, end = filename.split('..')
        assert len(start) == len(end)
        str_len = len(start)
        prefix = ""
        if "hdfs://" in url_short:
            prefix = "pipe:hdfs dfs -cat "
        for j in range(int(start), int(end)+1):
            urls.append(f"{prefix}{url_path}/{str(j).zfill(str_len)}{ext}")
    except:
        logging.warn(f"Failed processing {url_short}")
    return urls

def get_urls(train_webdataset_list):
    
    urls = []
    if "galaxyark" in train_webdataset_list:
        logging.info("added webdataset: galaxyark")
        urls = add_urls(urls, "/mnt/bd/mss-unlabeled/NewGalaxyArk/audio_16_44100hz_shard/{00000..00076}.tar")

    if "galaxyark_hdfs" in train_webdataset_list:
        logging.info("added webdataset: galaxyark_hdfs")
        urls = add_urls(urls, "hdfs://harunava/home/byte_speech_sv/data/hot_galaxy/new_4stem_16_44100_shard/{00000..00076}.tar")
        
    if "musdb18hq" in train_webdataset_list:
        logging.info("added webdataset: musdb18hq")
        urls = add_urls(urls, "/mnt/bd/mss-unlabeled/musdb18hq/4stem_full_44100_shard/{00000..00004}.tar")

    if "musdb18hq_hdfs" in train_webdataset_list:
        logging.info("added webdataset: musdb18hq_hdfs")           
        urls = add_urls(urls, "4stem_full_44100_shard/{00000..00004}.tar")
    
    if "epidemic_hdfs" in train_webdataset_list:
        logging.info("added webdataset: epidemic_hdfs")
        urls = add_urls(urls, "hdfs://harunava/home/byte_speech_sv/data/epidemic/4stem_clean_vocal_-32db_wav_16_44100hz_shard/{00000..00074}.tar")

    if "epi_inst_hdfs" in train_webdataset_list:
        logging.info("added webdataset: epi_inst_hdfs")
        urls = add_urls(urls, "hdfs://harunava/home/byte_speech_sv/data/epidemic/4stem_clean_vocal_-24db_wav_16_44100hz_shard/{00000..00019}.tar")
        urls = add_urls(urls, "hdfs://harunava/home/byte_speech_sv/data/epidemic/4stem_clean_vocal_-16db_wav_16_44100hz_shard/{00000..00009}.tar")
        urls = add_urls(urls, "hdfs://harunava/home/byte_speech_sv/data/epidemic/3stem_instrument_wav_16_44100hz_shard/{00000..00041}.tar")

    if "epidemic" in train_webdataset_list:
        logging.info("added webdataset: epidemic")
        urls = add_urls(urls, "/mnt/bd/mss-unlabeled/epidemic_wav/mss_full_44100hz_shard/{00000..00099}.tar")
        
    if "epidemicmp3_hdfs" in train_webdataset_list:
        logging.info("added webdataset: epidemic_mp3")
        urls = add_urls(urls, "hdfs://harunava/home/byte_speech_sv/data/epidemic/4stem_mp3_16_44100hz_shard/{00000..00077}.tar") 

    if "karaoke" in train_webdataset_list:
        logging.info("added webdataset: karaoke")
        urls = add_urls(urls, "/mnt/bd/mss-unlabeled/karaoke/18stem_clip_44100_shard/{00000..00503}.tar")

    if "tency500hq" in train_webdataset_list:
        logging.info("added webdataset: tency500hq")
        urls = add_urls(urls, "/mnt/bd/mss-unlabeled/tency500hq/18stem_full_44100_shard/{00000..00027}.tar")
    
    if "shutter" in train_webdataset_list:
        logging.info("added webdataset: shutter")
        urls = add_urls(urls, "/home/tiger/shutter_webdataset/{00000..00010}.tar")

    if "_test" in train_webdataset_list:
        logging.info("added webdataset: _test")
        urls = add_urls(urls, "/mnt/bd/mss-unlabeled/karaoke/18stem_clip_44100_shard/{00123..00123}.tar")

    return urls


def fix_hash(x):
    return int(hashlib.sha256(x.encode("utf-8")).hexdigest(), 16) % 10**8


# Initialize webdataset
class CommonWebDataset(wds.DataPipeline):
    def __init__(
        self,
        train_webdataset_list=[],
        chunk_len_s=8,
        sampling_rate=44100,
        task_type="4_stems",
        input_sources=["vocals", "bass", "drums", "other"], 
        seed=2023,
        shuffle=True,
        nodesplitter=wds.single_node_only,
        shard_buffer_size=10,
        sample_buffer_size=100,
        batch_size=2,
        resampled=True,
    ):
        super().__init__()
        self._train_webdataset_list = train_webdataset_list
        self.len_audio_sample = int(sampling_rate * chunk_len_s)

        if task_type == "4_stems":
            self.input_sources = MSS_4_STEM
            self.source_conversion = MSS_18_TO_4_STEM
        elif task_type == "9_stems":
            self.input_sources = MSS_9_STEM
            self.source_conversion = MSS_18_TO_9_STEM
        else:
            raise ValueError(f'task_type = {task_type}, should be "4_stems" or "9_stems"')

        self.input_sources = input_sources
        self.batch_size = batch_size

        audio_urls = get_urls(self._train_webdataset_list)

        if resampled:
            self.append(wds.ResampledShards(audio_urls, deterministic=False))
        else:
            self.append(wds.SimpleShardList(audio_urls, seed=seed))
            self.append(nodesplitter)
            self.append(wds.split_by_worker)
            if shuffle:
                self.append(wds.detshuffle(shard_buffer_size, seed=seed))
        self.append(wds.tarfile_to_samples())

        self.append(wds.decode())
        self.append(wds.map(self._process_audio))
        self.append(wds.to_tuple('audio', 'id'))
        self.append(wds.shuffle(sample_buffer_size, rng=random.Random(seed)))
        self.append(
            wds.batched(batch_size, collation_fn=self._collate)
        )

    def _get_audio_np_from_data(self, data, source):
        audio = data[f"{source}.npy"] # np array
        if audio.shape[-1] < self.len_audio_sample:
            audio = np.pad(
                audio,
                ((0, 0), (0, self.len_audio_sample - audio.shape[-1])),
                "constant",
            )
            start_idx = 0
        else:
            start_idx = random.randint(0, audio.shape[-1] - self.len_audio_sample)
        audio =  audio[..., start_idx : start_idx + self.len_audio_sample]
        if audio.dtype == np.int16:
            audio = (audio / 32768).astype(np.float32)    # np.int16 -> np.float32 

        return audio

    def _process_audio(self, data):

        if "audio.npy" in data:  # S x 2 x T
            audio = data["audio.npy"]

            if audio.shape[-1] < self.len_audio_sample:
                audio = np.pad(
                    audio,
                    ((0, 0), (0, 0), (0, self.len_audio_sample - audio.shape[-1])),
                    "constant",
                )
                start_idx = 0
            else:
                start_idx = random.randint(0, audio.shape[-1] - self.len_audio_sample)

            audio = audio[..., start_idx : start_idx + self.len_audio_sample]
            if audio.dtype == np.int16:
                audio = audio / 32768
            combined_audio = torch.from_numpy(audio).float()
        else:      

            combined_audio = [] # the last one must be "other"
            data_sources = [s.replace('.npy', '') for s in data if '.npy' in s]
            
            for source in self.input_sources: 
                audio = np.zeros((2, self.len_audio_sample), dtype=np.float32)
                
                for raw_source in data_sources:
                    if raw_source in self.source_conversion:
                        _source = self.source_conversion[raw_source]
                    else:
                        _source = raw_source
                    if _source == source:
                        audio += self._get_audio_np_from_data(data, raw_source)

                combined_audio.append(audio)
                
            combined_audio = np.stack(combined_audio, axis=0)  
            combined_audio = torch.from_numpy(combined_audio).float()  # torch.float32

        new_data = {}
        new_data["audio"] = combined_audio
        new_data["id"] = fix_hash(data["__key__"])

        return new_data

    def _collate(self, batch): # batch list of tuple ( tensor(4, 2, self.len_audio_sample), id)
        # target new_batch["vocals"] = (self.batch_size, 2, self.len_audio_sample)

        source_batch = {source: [] for source in self.input_sources}
        key_ids = []
        for audio, key_id in batch: 
            for i, source in enumerate(self.input_sources):
                source_batch[source].append(audio[i, :])
            key_ids.append(key_id)
        
        new_batch = {}
        for source in self.input_sources:
            new_batch[source] = torch.stack(source_batch[source])
        new_batch['id'] = torch.tensor(key_ids)

        #print(type(new_batch[source]), new_batch[source].shape, new_batch['id'])
        return new_batch

class MSSDataModule(pl.LightningDataModule):
    data_sample_rate = None
    def __init__(
        self,
        num_workers: int = 8,
        pin_memory: bool = True,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=None, num_workers=self.num_workers)



if __name__ == "__main__":
    from samantha.dataio.dataset import MultiIterableDataset
    import soundfile as sf

    train_dataset = CommonWebDataset(
        train_webdataset_list = ["musdb18hq_hdfs"],
        chunk_len_s=8,
        sampling_rate=44100,
        task_type="4_stems",
        seed=2023,
        shuffle=True,
        batch_size=2,
        resampled=True,
        sample_buffer_size=1,
    )

    def my_woker_init_fn(worker_id):
        torch.random.seed()
        np.random.seed()
        random.seed()

    train_dataset = MultiIterableDataset(
        datasets=[train_dataset],
        num_samples=1_000_000_000,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=1,
        worker_init_fn=my_woker_init_fn,
        persistent_workers=True,
    )

    os.makedirs("stem_audio_dumps", exist_ok=True)
    # Iterate over the dataset
    cnt = 0
    for i, sample in enumerate(train_dataloader):
        cnt += 1
        for source in sample:
            if source != "id":
                sf.write(f"stem_audio_dumps/{i}_{source}.wav", sample[source][0].numpy().T, 44100)

        if cnt == 10:
            break
