import itertools
import os
import json
from pathlib import Path
from recipes.l2v.datasets.tokenizers.phoneme_tokenizer import PhonemeTokenizer
from recipes.l2v.datasets.transforms.lyrics import LyricsTokenTransform
from torch.utils.data import Dataset, DataLoader
from recipes.l2v.datasets.lyrics import LyricsDataset

default_prompt_path = Path(__file__).absolute().parent/'inference_prompts/default.json'

class CoarseInferenceDataset(Dataset):
    def __init__(self, items, tokenizer=None):
        if tokenizer is None:
            self.tokenizer = PhonemeTokenizer(allow_unknown=False)
        else:
            self.tokenizer = tokenizer
        self.items = items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        lyrics_tokens = self.tokenizer(item['lyrics']) if 'lyrics' in item else None
        return {
            'lyrics': item.get('lyrics'),
            'lyrics_tokens': lyrics_tokens,
            'target_audio': item.get('target_audio'),
            'mulan_audio': item.get('mulan_audio'),
            'mulan_text': item.get('mulan_text'),
            'vocal_audio': item.get('vocal_audio'),
            'vocal_chroma': item.get('vocal_chroma')
        }
    
    @classmethod
    def default_dataset(cls, inference_type, prompt_path=None, max_items=16):
        if prompt_path is None or not os.path.exists(prompt_path):
            print('Prompt path not found. Using default', default_prompt_path)
            prompt_path = default_prompt_path
        with open(prompt_path, 'r') as f:
            prompts = json.load(f)
            input_lyrics = prompts['lyrics']
            input_prompts = prompts['mulan_text']
        if inference_type == 'text_prompt':
            return cls.default_lyrics_text_prompt(input_lyrics, input_prompts, max_items)
        if inference_type == 'audio_prompt':
            return cls.default_lyrics_audio_prompt(input_lyrics, input_prompts, max_items)

    @classmethod
    def default_lyrics_text_prompt(cls, input_lyrics, input_prompts, max_items=16):
        lyrics_prompt_pairs = list(itertools.product(input_lyrics, input_prompts))
        default_phoneme_predict_items = [ { 'lyrics': lyrics, 'mulan_text': mulan_text } for lyrics, mulan_text in lyrics_prompt_pairs ]
        if max_items is not None:
            default_phoneme_predict_items = default_phoneme_predict_items[:max_items]
        lyrics_tokenizer = PhonemeTokenizer(allow_unknown=False)
        predict_dataset = CoarseInferenceDataset(default_phoneme_predict_items, lyrics_tokenizer)
        return predict_dataset

    @classmethod
    def default_lyrics_audio_prompt(cls, input_lyrics, input_mulan_text, max_items=16):
        lyrics_tokenizer = PhonemeTokenizer(allow_unknown=False)

        # Local: 
        if os.path.exists('/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/webdataset/shards-0131.tar'):
            # url2index = {
            #     '/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/webdataset/shards-0000.tar': 
            #     '/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/webdataset/shards-0000.tar.index'
            # }
            url2index = {
                '/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/webdataset/shards-0131.tar': 
                '/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/webdataset/shards-0131.tar.index'
            }
        else:
            # Web:
            url2index = '/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/karaoke_valid.tar_to_index.tsv'

        sample_rate = 24_000
        duration = 10
        segment_transforms = [LyricsTokenTransform(lyrics_tokenizer, 150)]
        valid_ds = LyricsDataset(
            url2index,
            sample_rate=sample_rate,
            sample_duration=duration,
            segment_transforms=segment_transforms,
            audio_keys={ "target_audio": "full.mp3", "mulan_audio": "full.mp3", "vocal_audio": "vocal.mp3" },
            audio_format='mp3',
            max_num_segments=1, # Only return one segment per song for variety
        )
        it = iter(valid_ds)
        predict_items = [next(it) for i in range(max_items)]

        # Note: Shifted items is a hack to switch up target vocal conditioning
        shifted_predict_items = [item.copy() for item in predict_items[1:] + [predict_items[0]]]
        for idx, (item, shifted_item, lyrics, mulan_text) in enumerate(
            zip(predict_items, shifted_predict_items, itertools.cycle(input_lyrics), itertools.cycle(input_mulan_text))
        ):
            item['lyrics'] = lyrics
            item['mulan_text'] = mulan_text
            item['vocal_audio'] = shifted_item['vocal_audio']
            item['vocal_chroma'] = shifted_item.get('vocal_chroma')

        predict_dataset = CoarseInferenceDataset(predict_items, lyrics_tokenizer)
        return predict_dataset
    

def create_datamodule(predict_dataset: CoarseInferenceDataset, batch_size=8, lyrics_max_seq_len=150, audio_max_seq_len=24_000*10):
    from recipes.musiclm.datamodules.lit_datamodule import DataModule
    from recipes.l2v.datasets.lyrics import LyricsCollator
    lyrics_collate = LyricsCollator(
        lyrics_padding_value=predict_dataset.tokenizer.pad_id,
        lyrics_max_seq_len=lyrics_max_seq_len,
        audio_max_seq_len=audio_max_seq_len
    )
    predict_dataloader = DataLoader(
        dataset=predict_dataset, 
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        collate_fn=lyrics_collate
    )
    pl_datamodule = DataModule(predict_dataloader=predict_dataloader)
    return pl_datamodule
