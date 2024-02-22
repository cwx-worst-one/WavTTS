import json
import os
import random

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import time
import webdataset as wds
from recipes.mulan.dataset.utils import *
from samantha.dataio.webdataset.extension import IndexedWebDataset
import json
import librosa
import io
from samantha.dataio.parquet.parquet_dataset import ParquetDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# /mnt/bn/audio-diffusion/data/qmusic/
def pattern_apply(text_content):
    try:
        pattern = r'"text":\s*"([^"]+)"'
        match_res = re.search(pattern, text_content)
        if match_res:
            extracted_text = match_res.group(1)
        else:
            return ""
    except:
        print("wrong text!")
        return ""
    return extracted_text

# for item in d:
#     metadata = json.loads(item["meta"])
#     print(metadata)
#     print("gpt_text:", gpt_text)
#     print("artist_name:", artist_name_res)
#     print("track_name:", track_name_res)
#     print("album_name:", album_name_res)
# print("Total items:", cnt)


def return_self(x):
    return x
def process_audio_pq(data, segment = None):
    audio, _ = librosa.load(io.BytesIO(data["wav"]), sr = None)
    data["audio.npy"] = audio
    del data["wav"]
    data = process_audio(data, segment)
    return data

class CMDatasetPar(IterableDataset):
    def __init__(self, name="qq", mode="train", tokenizer_name="chinese",**kwargs):
        if tokenizer_name =="chinese":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer_name == "multiligual":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.name = name
        dataset_id = 401
        self.dataset = (
            ParquetDataset(dataset_id, handler=wds.warn_and_continue, **kwargs)
            .map(process_audio_pq)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        metadata = json.loads(data["meta"])
        if self.name == "qq":
            text_res = metadata.get('gpt_text', "")
            #print(text_res)
            # artist_name = metadata.get('artist_name', "")
            # track_name = metadata.get('track_name', "")
            # album_name = metadata.get('album_name', "")
            # artist_name_res = pattern_apply(artist_name)
            # track_name_res = pattern_apply(track_name)
            # album_name_res = pattern_apply(album_name)
            theme= re.findall(r'情绪：\s*([^\n]+)', text_res)
            data["text"] = theme
            #print(theme)
            # data["text"] = ". ".join([artist_name_res, track_name_res, album_name_res, gpt_text])
            data["music_id"] = fix_hash(data["__key__"])
            return data  
        else:
            return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":
    dataset = CMDatasetPar(name="qq", mode="train")
    phrase_frequency = {}
    cnt = 0
    for item in dataset:
        cnt += 1
        # print(item.keys())
        # print(item["audio"])
        #print(item["text"])
        try:
            vocal = item["text"][0]
            print(vocal)
            # 使用 "、" 分割输入字符串
            phrases = vocal.split('、')

            # 遍历分割后的短语列表
            for phrase in phrases:
                # 去除可能的前后空格
                phrase = phrase.strip()
                
                # 如果短语不在字典中，添加并初始化词频为1，否则增加词频
                if phrase in phrase_frequency:
                    phrase_frequency[phrase] += 1
                else:
                    phrase_frequency[phrase] = 1
        except:
            continue

        # 打印生成的字典
        #print(phrase_frequency)
        if cnt == 1000:
            break
    file_name = "vocal_counts.json"

    # 使用 json.dump() 将字典写入文件
    with open(file_name, "w") as file:
        json.dump(phrase_frequency, file)

    print(f"字典已保存到文件 {file_name}")
            #breakpoint()
    sorted_instrument_dict = dict(sorted(phrase_frequency.items(), key=lambda item: item[1], reverse=True))

    # 获取前20个词频最高的条目
    top_20 = list(sorted_instrument_dict.items())[:30]

    # 打印前20个词频最高的条目
    for item in top_20:
        print(f"{item[0]}: {item[1]}")