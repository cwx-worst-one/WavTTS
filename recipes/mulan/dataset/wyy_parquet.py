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
import ast
from typing import Dict, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_audio_pq(data, segment = None):
    audio, _ = librosa.load(io.BytesIO(data["wav"]), sr = None)
    data["audio.npy"] = audio
    del data["wav"]
    data = process_audio(data, segment)
    return data

def process_audio_segments(data, metadata):
    audio, _ = librosa.load(io.BytesIO(data["wav"]), sr = None)
    del data["wav"]

    if audio.dtype == np.int16:
        audio = (audio / 32768.0).astype("float32")
    if len(audio.shape) == 1:
        audio = audio[None, :]

    music_len = 24000 * 10
    if audio.shape[-1] < music_len:
        audio = np.pad(audio, ((0, 0), (0, music_len - audio.shape[-1])), "constant")
        start_idx = 0
    elif 'deepchorus' in metadata:
        segments = metadata['deepchorus']['segments']
        audio_duration = audio.shape[-1]
        start_candidates = []
        for s in segments:
            label = s['label']
            start, end = s['interval']
            start_idx = int(start * 24000)
            end_idx = int(min(end * 24000, audio_duration))
            if label in ['chorus', 'verse', 'bridge', 'inst'] and (start_idx + music_len) <= audio_duration:
                random_section_start = random.randint(start_idx, max(start_idx, end_idx - music_len))
                start_candidates.append(random_section_start)
        if start_candidates:
            start_idx = random.choice(start_candidates)
        else:
            start_idx = random.randint(0, max(0, audio.shape[-1] - music_len))
    elif audio.shape[-1] > music_len * 4: # trim intro / outro
        start_idx = random.randint(music_len, audio.shape[-1] - music_len * 2)
    else:
        start_idx = random.randint(0, max(0, audio.shape[-1] - music_len))

    audio = torch.from_numpy(audio[..., start_idx : start_idx + music_len]).float()
    data["audio"] = audio
    return data


WYY_TAG_ZH_TO_EN = {    # totally 75 tags from wyy raw meta
    # 时间相关
    '午休': 'Nap',
    '下午茶': 'Afternoon Tea',
    '夜晚': 'Night',
    '清晨': 'Morning',
    # 情感相关
    '性感': 'Sexy',
    '快乐': 'Happy',
    '感动': 'Touching',
    '浪漫': 'Romantic',
    '孤独': 'Lonely',
    '伤感': 'Sad',
    '思念': 'Missing',
    '兴奋': 'Excited',
    '怀旧': 'Nostalgic',
    '安静': 'Quiet',
    '治愈': 'Healing',
    '放松': 'Relaxing',
    '清新': 'Fresh',
    # 年代相关
    '70后': '70s',
    '80后': '80s',
    '90后': '90s',
    '00后': '00s',
    # 语言相关
    '日语': 'Japanese',
    '韩语': 'Korean',
    '粤语': 'Cantonese',
    '华语': 'Mandarin',
    '小语种': 'Minor Languages',
    '英伦': 'Britpop',
    # 类型相关
    '拉丁': 'Latin',
    '综艺': 'Variety Show',
    '民谣': 'Chinese Folk',
    '欧美': 'Western',
    '民族': 'Ethnic',
    '电子': 'Electronic',
    '爵士': 'Jazz',
    '蓝调': 'Blues',
    '雷鬼': 'Reggae',
    '金属': 'Metal',
    '另类/独立': 'Alternative/Indie',
    '摇滚': 'Rock',
    '朋克': 'Punk',
    'R&B/Soul': 'R&B/Soul',
    'New Age': 'New Age',
    '流行': 'Pop',
    '世界音乐': 'World Music',
    '影视原声': 'Soundtrack',
    '音乐剧': 'Musical',
    '轻音乐': 'Light Music',
    '古典': 'Classical',
    '古风': 'Chinese Tradition',
    'Bossa Nova': 'Bossa Nova',
    '后摇': 'Post Rock',
    '乡村': 'Country',
    '经典': 'Classics',
    # 场景相关
    '运动': 'Sports',
    '散步': 'Walk',
    '地铁': 'Subway',
    '旅行': 'Travel',
    '驾车': 'Driving',
    '学习': 'Study',
    '工作': 'Work',
    '酒吧': 'Bar',
    'KTV': 'KTV',
    '校园': 'Campus',
    '网络歌曲': 'Internet Songs',
    # 音乐制作相关
    'ACG': 'ACG',
    '器乐': 'Instrumental',
    '吉他': 'Guitar',
    '钢琴': 'Piano',
    '翻唱': 'Cover',
    '舞曲': 'Dance',
    '说唱': 'Rap',
    '游戏': 'Game',
    # 其他
    '官方': 'Official',
    '榜单': 'Chart',
    '儿童': 'Children',
}

def parse_freeform_text_short_wyy_optional(meta: Dict) -> Optional[str]:
    """meta.raw.tags, meta.raw.category"""
    tags = meta.get("raw", {}).get("tags", "[]")
    category = meta.get("raw", {}).get("category", "")
    tags = ast.literal_eval(tags)
    keywords = set(tags)
    keywords.add(category)
    keywords = list(keywords)
    # not_found = [x for x in keywords if x not in WYY_TAG_ZH_TO_EN]
    # if not_found:
    #     print('Not found', not_found)
    keywords = [WYY_TAG_ZH_TO_EN[x] if x in WYY_TAG_ZH_TO_EN else "" for x in keywords]
    keywords = [x for x in keywords if x]
    if random.random() < 0.5:
        keywords = [x.lower() for x in keywords if x]
    random.shuffle(keywords)
    freeform_text_short = ", ".join(keywords)
    return freeform_text_short if freeform_text_short else None

def parse_music_tagging(meta: Dict) -> Optional[str]:
    genre = meta['music_tagging']['Genre20']['result']
    moods = meta['music_tagging']['Mood']['result']
    theme = meta['music_tagging']['Theme']['result']
    keywords = [genre]+ moods + theme
    if random.random() < 0.5:
        keywords = [x.lower() for x in keywords if x]
    random.shuffle(keywords)
    return ", ".join(keywords) if keywords else None

class WYYDataset(IterableDataset):
    def __init__(self, name="wyy", mode="train", text_pick = "random", dataset_id=317, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_general")
        self.name = name
        self.text_pick = text_pick
        self.dataset = (
            ParquetDataset(dataset_id, handler=wds.warn_and_continue, **kwargs)
            .map(self._process_audio)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode))
        )
    def _process_audio(self, data):
        metadata = json.loads(data["meta"])
        return process_audio_segments(data, metadata)


    def _process_text(self, data):
        metadata = json.loads(data["meta"])
        if random.random() < 0.8:
            text = parse_freeform_text_short_wyy_optional(metadata)
        else:
            text = parse_music_tagging(metadata)
        if text is None: return None
        data["text"] = text
        data["music_id"] = fix_hash(data["__key__"])
        return data

    def __iter__(self):
        return iter(self.dataset)

def sample_pct(arr, dropout=0.5, min_examples=1):
    random.shuffle(arr)
    if len(arr) * (1 - dropout) <= min_examples:
        return arr
    return [a for idx, a in enumerate(arr) if random.random() >= dropout]

if __name__ == "__main__":
    dataset = WYYDataset(name="wyy", mode="train")
    cnt = 0
    for item in dataset:
        cnt += 1
        print(item.keys())
        print(item["audio"])
        print(item["text"])
        if cnt == 5:
            break
        #breakpoint()
        
