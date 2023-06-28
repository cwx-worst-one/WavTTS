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
    codecs = [batch[0] for batch in batches]
    texts = [batch[1] for batch in batches]
    max_codec_len = max([codec.shape[0] for codec in codecs])
    max_text_len = max([text.shape[0] for text in texts])

    n_code = codecs[0].shape[1]
    bs = len(codecs)
    new_codecs = np.zeros(shape=[bs, max_codec_len + 1, n_code], dtype=np.int32)
    new_texts = np.zeros(shape=[bs, max_text_len], dtype=np.int32)
    codec_lens = np.zeros(shape=[bs,], dtype=np.int32)
    text_lens = np.zeros(shape=[bs,], dtype=np.int32)

    for i, codec in enumerate(codecs):
        l = codec.shape[0]
        new_codecs[i, 0:l, :] = codec
        new_codecs[i, l, :] = 1024
        codec_lens[i] = l + 1
    for i, text in enumerate(texts):
        l = text.shape[0]
        new_texts[i, 0:l] = text
        text_lens[i] = l

    new_codecs = torch.from_numpy(new_codecs)[:, :, 0]
    new_texts = torch.from_numpy(new_texts)
    codec_lens = torch.from_numpy(codec_lens)
    text_lens = torch.from_numpy(text_lens)

    return new_codecs, new_texts, codec_lens, text_lens


class CLVPDataset(IterableDataset):
    def __init__(
        self,
        meta_paths,
        tokenizer_cls,
    ):
        self.tokenizer = tokenizer_cls()
        #self.min_dur = min_dur
        #self.max_dur = max_dur

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
            .shuffle(100000)        # buffer size
        )

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def _normalize(self, item):
        key = item['__key__']
        wav_id = item['wav_id.npy'] # [t, n_code]
        text = item['text'].lower() + '</s>'
        text_id = self.tokenizer(text, return_tensors='np')['input_ids'][0]
        if wav_id.shape[0] > 20 * 80:
            return None
        return wav_id, text_id
