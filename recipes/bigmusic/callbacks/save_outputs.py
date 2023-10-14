import pytorch_lightning as pl
from typing import Any
import json
from pathlib import Path
import os
import textwrap
import shutil
from recipes.musiclm.inference.utils import slugify, save_wav, generate_hash, format_name, load_wav

class SaveOutputsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.total_items = 0

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        output_dir = pl_module.extra_params.output_dir
        sample_rate = pl_module.extra_params.sample_rate
        save_batch_outputs(outputs, batch, output_dir=output_dir, sample_rate=sample_rate, sample_round=dataloader_idx, index_offset=self.total_items)
        self.total_items += outputs['generated_audio_tensor'].shape[0]
    
def format_lyrics_and_style(style_text, lyrics=None):
    if style_text is None and lyrics is None: # gt case
        return ""
    if style_text is None: # dry vocal case
        return format_name(lyrics)
    if lyrics is None:
        return format_name(style_text) # instrumental use case
    lyrics_formated = slugify(lyrics) # this function was already there for MusicLM
    text_formated = slugify(style_text)
    text_combined = lyrics_formated + "_" + text_formated
    text_encoded = generate_hash(text_combined) # this function was already there for MusicLM
    name_formatted = text_formated[:32] + "_" + lyrics_formated[:96] + "_" + text_encoded[:4]
    return name_formatted

def save_batch_outputs(outputs, batch, output_dir, sample_rate, sample_round=0, index_offset=0):
    conditions = batch['conditions']
    lyrics = batch.get('lyrics')
    lyrics_normalized_text = batch.get('lyrics_normalized_text')
    prompts = batch.get('style_text')
    categories = batch.get('category')
    style_audio = batch.get('style_audio')
    vocal_audio = batch.get('vocal_audio')
    indexes = batch.get('index')
    metadatas = outputs.get('metadata')
    wavs = outputs['generated_audio']
    
    for i, wav in enumerate(wavs):
        if categories is not None and categories[i]:
            wav_dir = os.path.join(output_dir, str(categories[i]))
        else:
            wav_dir = output_dir
        os.makedirs(wav_dir, exist_ok=True)
        file_name = ""
        absolute_idx =  i + index_offset
        lyrics_str = lyrics[i] if 'lyrics_tokens' in conditions else None
        lyrics_normalized_str = lyrics_normalized_text[i] if 'lyrics_tokens' in conditions and lyrics_normalized_text else None
        style_text = prompts[i] if 'style_text' in conditions else None
        if indexes is not None:
            file_name = indexes[i]
        else:
            file_name = f"{absolute_idx:03d}_{format_lyrics_and_style(style_text, lyrics_str)}"
        wav_fp = os.path.join(wav_dir, f"{file_name}.generated.wav")
        print(f"[Saving] {wav_fp}")
        save_wav(wav.cpu().float(), wav_fp, sr=sample_rate)
        if style_audio is not None:
            input_wav_fp = os.path.join(wav_dir, f"{file_name}.style_audio.wav")
            save_wav(style_audio[i].cpu().float(), input_wav_fp, sr=sample_rate)

        if vocal_audio is not None:
            input_vocals_fp = os.path.join(wav_dir, f"{file_name}.vocal_audio.wav")
            save_wav(vocal_audio[i].cpu().float(), input_vocals_fp, sr=sample_rate)

        meta_fp = os.path.join(wav_dir, f"{file_name}.metadata.json")
        metadata = metadatas[i] if metadatas is not None else {}
        metadata = {
            **metadata,
            'lyrics': lyrics_str,
            'lyrics_normalized_text': lyrics_normalized_str,
            'style_text': style_text,
            'conditions': conditions,
            'index': {
                'round': sample_round,
                'absolute_idx': absolute_idx,
                'batch_idx': i
            }
        }
        print('Saving metadata', metadata)
        with open(meta_fp, 'w') as f:
            json.dump(metadata, f, indent=2)


class SaveVideoCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        save_video(output_dir, output_dir)

def default_format_video_text(metadata):
    lyrics = metadata['lyrics']
    index = metadata['index']['absolute_idx']
    lyrics = lyrics.encode('ascii', 'ignore').decode('ascii') # TODO: utf-8
    lyrics_list = textwrap.wrap(lyrics, 40, break_long_words=False)
    style_text = metadata['style_text']
    return f'{index}: {style_text}\n\n\n' + '\n\n'.join(lyrics_list)

def save_video(input_results_dir, output_video_dir, format_video_text_fn=default_format_video_text, remove_segments=True):
    colors = ["green", "blue", "brown"]
    output_video_dir = Path(output_video_dir)
    output_video_dir_tmp = output_video_dir/'tmp'
    output_video_dir_tmp.mkdir(exist_ok=True, parents=True)
    generated_output_fps = list(Path(input_results_dir).glob('**/*.generated.wav'))
    for idx, generated_output_fp in enumerate(generated_output_fps):
        audio_fp = generated_output_fp
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        
        output_video_fp = output_video_dir_tmp/generated_output_fp.with_suffix('.mp4').name
        output_text_fp = output_video_dir_tmp/generated_output_fp.with_suffix('.txt').name
        with open(metadata_fp, 'r') as f:
            metadata = json.load(f)
        with open(output_text_fp, 'w') as f:
            f.write(format_video_text_fn(metadata))
        color = colors[idx % len(colors)]
        cmd = f'ffmpeg -y -f lavfi -i color=c={color}:s=800x800:d=0.5 -i {audio_fp} -c:a aac -vf "drawtext=fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile={output_text_fp}" {output_video_fp}'
        os.system(cmd)

    # concat output videos
    video_output_fp = output_video_dir/f"vocal_music_demo.mp4"
    cmd_concat = f"cd {output_video_dir_tmp} && find *.mp4 | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt; ffmpeg -f concat -i fl.txt -c copy output.mp4; rm fl.txt"
    os.system(cmd_concat)
    (output_video_dir_tmp/"output.mp4").rename(video_output_fp)
    if remove_segments:
        shutil.rmtree(output_video_dir_tmp)
    return video_output_fp