import torch
import numpy as np
import librosa
import webdataset as wds
import ffmpeg


from torch.utils.data import IterableDataset


def ffmpeg_read_audio(audio_bin, sample_rate=24000):
    st = time.time()
    seg_bin, err = ffmpeg.input("pipe:").output("pipe:", loglevel="error", format="s16le", ar=sample_rate).run(input=audio_bin, quiet=True)
    return (np.frombuffer(seg_bin, dtype="int16") / 32768.0).astype(np.float32)


def collate_fn(batches):
    max_16k_len = max([batch[0].shape[-1] for batch in batches])
    max_24k_len = max([batch[1].shape[-1] for batch in batches])
    max_text_len = max([batch[2].shape[-1] for batch in batches])

    b = len(batches)
    wavs_16k = np.zeros(shape=[b, max_16k_len])
    wavs_24k = np.zeros(shape=[b, max_24k_len])
    texts = np.zeros(shape=[b, max_text_len])

    wav_16k_lens = np.zeros(shape=[b,], dtype=np.int64)
    wav_24k_lens = np.zeros(shape=[b,], dtype=np.int64)
    text_lens = np.zeros(shape=[b,], dtype=np.int64)

    for i, batch in enumerate(batches):
        wav_16k, wav_24k, text = batch

        wavs_16k[i][0:wav_16k.shape[-1]] = wav_16k
        wavs_24k[i][0:wav_24k.shape[-1]] = wav_24k
        texts[i][0:text.shape[-1]] = text

        wav_16k_lens[i] = wav_16k.shape[-1]
        wav_24k_lens[i] = wav_24k.shape[-1]
        text_lens[i] = text.shape[-1]

    wavs_16k = (wavs_16k * 32768.0).astype(np.int32) # save as int32 to prevent deepspeed fp16 transform
    wavs_24k = (wavs_24k * 32768.0).astype(np.int32) # save as int32 to prevent deepspeed fp16 transform
    wavs_16k = torch.from_numpy(wavs_16k)
    wavs_24k = torch.from_numpy(wavs_24k)
    texts = torch.from_numpy(texts).long()

    wav_16k_lens = torch.from_numpy(wav_16k_lens).long()
    wav_24k_lens = torch.from_numpy(wav_24k_lens).long()
    text_lens = torch.from_numpy(text_lens).long()

    return wavs_16k, wavs_24k, texts, wav_16k_lens, wav_24k_lens, text_lens


class WDSDataset(IterableDataset):
    def __init__(
        self,
        meta_paths,
        tokenizer_cls,
        min_dur=3,
        max_dur=30,
    ):
        self.min_dur = min_dur
        self.max_dur = max_dur

        urls = []
        for meta_path in meta_paths:
            if 'tar_list' in meta_path:
                with open(meta_path, 'r') as f:
                    lines = [l.strip() for l in f]
                for l in lines:
                    urls.append("pipe: hdfs dfs -cat {}".format(l))
            else:
                urls.append("pipe: hdfs dfs -cat {}".format(meta_path))

        self.dataset = (
            wds.WebDataset(urls=urls, resampled=True)
            .decode()             # 根据已知的后缀名或类型进行解码
            .map(self._normalize) # 应用自定义处理函数
            .shuffle(2000)        # buffer size
        )

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def _normalize(self, item):
        key = item['__key__']
        # TODO: 以后肯定要改，这么存在超大数据量下根本存不下
        wav_24k = item['wav.npy']
        text = item['text_id.npy']

        dur = wav_24k.shape[-1] / 24000
        # skip too short or too long sentences
        if dur < self.min_dur:
            return None
        elif dur > self.max_dur:
            seg_size = int(self.max_dur * 24000)
            beg = np.random.randint(low=0, high=wav_24k.shape[-1] - seg_size + 1)
            wav_24k = wav_24k[beg:beg + seg_size]

        if wav_24k.dtype == np.int16:
            wav_24k = wav_24k / 32768.0
        elif wav_24k.dtype in [np.float32, np.float64]:
            wav_24k = wav_24k
        else:
            raise Exception("Not supported dtype: {}".format(wav_24k.dtype))

        wav_24k = wav_24k / max(0.001, np.max(np.abs(wav_24k))) * 0.95
        wav_16k = librosa.resample(y=wav_24k, orig_sr=24000, target_sr=16000)

        return wav_16k, wav_24k, text


class SoundStormWDSDataset(IterableDataset):
    def __init__(
        self,
        meta_paths,
        tokenizer_cls,
        min_dur=3,
        max_dur=30,
    ):
        self.min_dur = min_dur
        self.max_dur = max_dur

        urls = []
        for meta_path in meta_paths:
            if 'tar_list' in meta_path:
                with open(meta_path, 'r') as f:
                    lines = [l.strip() for l in f]
                for l in lines:
                    urls.append("pipe: hdfs dfs -cat {}".format(l))
            else:
                urls.append("pipe: hdfs dfs -cat {}".format(meta_path))

        self.dataset = (
            wds.WebDataset(urls=urls, resampled=True)
            .decode()             # 根据已知的后缀名或类型进行解码
            .map(self._normalize) # 应用自定义处理函数
            .shuffle(2000)        # buffer size
        )

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def _normalize(self, item):
        key = item['__key__']
        if 'singing' in key or 'Music' in key:
            return None
        if 'npy' in item:
            wav_24k = item['npy']
        elif 'wav.npy' in item:
            wav_24k = item['wav.npy']
        else:
            raise Exception("No audio is found")

        dur = wav_24k.shape[-1] / 24000
        # skip too short or too long sentences
        if dur < self.min_dur or dur > self.max_dur:
            return None

        if wav_24k.dtype == np.int16:
            wav_24k = wav_24k / 32768.0
        elif wav_24k.dtype in [np.float32, np.float64]:
            wav_24k = wav_24k
        else:
            raise Exception("Not supported dtype: {}".format(wav_24k.dtype))

        wav_24k = wav_24k / max(0.001, np.max(np.abs(wav_24k))) * 0.95
        wav_16k = librosa.resample(y=wav_24k, orig_sr=24000, target_sr=16000)

        text = np.random.randint(low=0, high=1024, size=[10,])

        return wav_16k, wav_24k, text
