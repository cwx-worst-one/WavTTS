import pytorch_lightning as pl
import torch
from typing import Any
import json
from pathlib import Path
import os
import textwrap
import shutil
from recipes.musiclm.inference.utils import slugify, save_wav, generate_hash, format_name, load_wav
from collections import defaultdict
from recipes.bigmusic.utils.format_utils import update_json
import numpy as np
from recipes.musiclm.utils.dist import local_zero_first


class SaveOutputsCallback(pl.Callback):
    def __init__(
        self,
        beam_size=1,
        samples_to_save=1,
        save_style_audio=True,
        save_mp3=False,
        save_semantic_tokens=False,
    ):
        super().__init__()
        self.total_items = 0
        self.beam_size = beam_size
        self.samples_to_save = samples_to_save
        self.save_style_audio = save_style_audio
        self.save_mp3 = save_mp3
        self.save_semantic_tokens = save_semantic_tokens

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
        output_paths = save_batch_outputs(
            outputs,
            batch,
            output_dir=output_dir,
            sample_rate=sample_rate,
            sample_round=dataloader_idx,
            index_offset=self.total_items,
            beam_size=self.beam_size,
            samples_to_save=self.samples_to_save,
            save_style_audio=self.save_style_audio,
            save_mp3=self.save_mp3,
            save_semantic_tokens=self.save_semantic_tokens,
        )
        num_items = outputs['generated_audio_tensor'].shape[0] // self.beam_size
        self.total_items += num_items
        with open(Path(output_dir)/'inference_params.json', 'w') as f:
            json.dump(pl_module.extra_params, f, indent=2)

        # save output paths so other callbacks can run metrics on audio
        if 'output_paths' in pl_module.extra_params:
            pl_module.extra_params['output_paths'].extend(output_paths)
        else:
            pl_module.extra_params['output_paths'] = output_paths


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


def save_batch_outputs(
    outputs,
    batch,
    output_dir,
    sample_rate,
    sample_round=0,
    index_offset=0,
    beam_size=1,
    samples_to_save=1,
    save_style_audio=True,
    save_mp3=False,
    save_semantic_tokens=False,
):
    conditions = batch['conditions']
    index = batch.get('index')
    lyrics = batch.get('lyrics')
    lyrics_normalized_text = batch.get('lyrics_normalized_text')
    prompts = batch.get('style_text')
    style_categories = batch.get('style_category')
    structures = batch.get('structure')
    categories = batch.get('category')
    style_audio = batch.get('style_audio')
    vocal_audio = batch.get('vocal_audio')   
    metadatas = outputs.get('metadata')
    wavs = outputs['generated_audio']
    semantic_tokens = outputs.get('generated_semantic_tokens')

    output_paths = []
    
    for i, wav in enumerate(wavs):
        prompt_idx = i // beam_size
        beam_idx = i % beam_size
        if beam_idx >= samples_to_save:
            continue
        ii = prompt_idx if sample_round == 0 else prompt_idx // sample_round
        if categories is not None and categories[prompt_idx]:
            wav_dir = os.path.join(output_dir, categories[ii])
        else:
            wav_dir = output_dir
        os.makedirs(wav_dir, exist_ok=True)
        file_name = ""
        absolute_idx =  prompt_idx + index_offset
        lyrics_str = lyrics[ii] if 'lyrics_tokens' in conditions else None
        lyrics_normalized_str = lyrics_normalized_text[ii] if 'lyrics_tokens' in conditions and lyrics_normalized_text else None        
        style_text = prompts[ii] if 'style_text' in conditions else None
        style_category = style_categories[ii] if 'style_category' in conditions and style_categories else None
        structure = structures[ii] if 'structure' in conditions else None
        if index is None:
            file_name = f"{absolute_idx:03d}_{format_lyrics_and_style(style_text, lyrics_str)}"
        else:
            file_name = index[ii]
        wav_file_name = file_name
        if sample_round > 0:
            wav_file_name += ("_r" + str(prompt_idx % sample_round))
        if samples_to_save > 1 and beam_size > 1:
            wav_file_name += f".{beam_idx}"
            
        wav_fp = os.path.join(wav_dir, f"{wav_file_name}.generated.wav")
        print(f"[Saving] {wav_fp}")
        save_wav(wav.cpu().float(), wav_fp, sr=sample_rate, save_mp3=save_mp3)
        if save_style_audio and style_audio is not None and beam_idx == 0:
            input_wav_fp = os.path.join(wav_dir, f"{file_name}.style_audio.wav")
            save_wav(style_audio[ii].cpu().float(), input_wav_fp, sr=sample_rate, save_mp3=save_mp3)

        if vocal_audio is not None and beam_idx == 0:
            input_vocals_fp = os.path.join(wav_dir, f"{file_name}.vocal_audio.wav")
            save_wav(vocal_audio[ii].cpu().float(), input_vocals_fp, sr=sample_rate, save_mp3=save_mp3)

        if save_semantic_tokens and semantic_tokens is not None:
            semantic_tokens_fp = os.path.join(wav_dir, f"{wav_file_name}.semantic_tokens.pt")
            torch.save(semantic_tokens[i], semantic_tokens_fp)

        meta_fp = os.path.join(wav_dir, f"{wav_file_name}.metadata.json")
        metadata = metadatas[i] if metadatas is not None else {}
        metadata = {
            **metadata,
            'lyrics': lyrics_str,
            'lyrics_normalized_text': lyrics_normalized_str,
            'style_text': style_text,
            'style_category': style_category,
            'structure': structure,
            'conditions': conditions,
            'index': {
                'round': sample_round,
                'absolute_idx': absolute_idx,
                'batch_idx': ii,
                'beam_idx': beam_idx,
            }
        }
        print('Saving metadata', metadata)
        with open(meta_fp, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        output_paths.append(wav_fp)
    return output_paths

class SaveVideoCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        with local_zero_first():
            if trainer.is_global_zero:
                save_video(output_dir, output_dir)


class NormVolumeCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        sample_rate = pl_module.extra_params.sample_rate
        if 'output_paths' in pl_module.extra_params:
            generated_output_fps = pl_module.extra_params.output_paths
        else:
            output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))

        for idx, generated_output_fp in enumerate(generated_output_fps):
            print ("normalize volume for: ",  generated_output_fp)
            command = "ffmpeg-normalize '%s' -t %d -ext wav -ar %d -o '%s' -f" % (generated_output_fp, -16, sample_rate, generated_output_fp)
            os.system(command)


def format_video_text(metadata, max_width=50):
    index = metadata['index']['absolute_idx']
    style_text = ""
    if metadata.get("style_text") is not None:
        style_text = metadata.get('style_text')
        style_text = '\n'.join(textwrap.wrap(style_text, max_width, break_long_words=False))
    elif metadata.get("style_category") is not None:
        style_text = metadata.get('style_category')
        style_text = '\n'.join(textwrap.wrap(style_text, max_width, break_long_words=False))
    video_text = f'{index}: {style_text}\n\n'

    lyrics = metadata.get('lyrics')
    if lyrics is not None:
        # Respect natural linebreaks
        lyrics = lyrics.split("\n")

        lyrics_list = []
        for x in lyrics:
            for wrap_idx, y in enumerate(textwrap.wrap(x, max_width, break_long_words=False)):
                if wrap_idx > 0:
                    lyrics_list.append("\t")
                lyrics_list.append(y)
                lyrics_list.append("\n")
        video_text += ''.join(lyrics_list)
    return video_text

def default_format_video_text(metadata):
    # short text
    fontsize, max_width, line_spacing = 26, 50, 14
    video_text = format_video_text(metadata, max_width)
    num_lines = len(video_text.split("\n"))
    if num_lines < 18:
        return video_text, fontsize, line_spacing
    
    # long text
    fontsize, max_width, line_spacing = 20, 60, 4
    video_text = format_video_text(metadata, max_width)
    num_lines = len(video_text.split("\n"))
    if num_lines < 36:
        return video_text, fontsize, line_spacing
    
    # really long text
    fontsize, max_width, line_spacing = 16, 80, 2
    video_text = format_video_text(metadata, max_width)
    num_lines = len(video_text.split("\n"))
    if num_lines < 44:
        return video_text, fontsize, line_spacing
    
    fontsize, max_width, line_spacing = 11, 90, 0
    video_text = format_video_text(metadata, max_width)
    num_lines = len(video_text.split("\n"))
    return video_text, fontsize, line_spacing

def save_video(input_results_dir, output_video_dir, format_video_text_fn=default_format_video_text, remove_segments=True):
    colors = ["green", "blue", "brown"]
    output_video_dir = Path(output_video_dir)
    output_video_dir_tmp = output_video_dir/'tmp'
    output_video_dir_tmp.mkdir(exist_ok=True, parents=True)

    generated_output_fps = list(Path(input_results_dir).glob('**/*.generated.wav'))

    mp3_fps = list(Path(input_results_dir).glob('**/*.generated.wav.mp3'))
    if len(mp3_fps) > 0 and len(generated_output_fps) == 0:
        raise Exception('Save video does not support mp3 outputs. Encoding to video causes artifacts')

    for idx, generated_output_fp in enumerate(generated_output_fps):
        audio_fp = generated_output_fp
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        
        output_video_fp = output_video_dir_tmp/generated_output_fp.with_suffix('.mp4').name
        output_text_fp = output_video_dir_tmp/generated_output_fp.with_suffix('.txt').name
        with open(metadata_fp, 'r') as f:
            metadata = json.load(f)
        with open(output_text_fp, 'w', encoding='utf-8') as f:
            video_text, fontsize, line_spacing = format_video_text_fn(metadata)
            f.write(video_text)
        color = colors[idx % len(colors)]
        fontfile = "/usr/share/fonts/truetype/arphic/ukai.ttc"
        cmd = f'ffmpeg -y -v 0 -f lavfi -i color=c={color}:s=800x800:d=0.5 -i {audio_fp} -c:a aac -vf "drawtext=fontfile={fontfile}:fontsize={fontsize}:line_spacing={line_spacing}:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile={output_text_fp}" {output_video_fp} -v 0'
        os.system(cmd)

    # concat output videos
    video_output_fp = output_video_dir/f"vocal_music_demo.mp4"
    cmd_concat = f"cd {output_video_dir_tmp} && find *.mp4 | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt; ffmpeg -v 0 -f concat -i fl.txt -c copy output.mp4; rm fl.txt"
    os.system(cmd_concat)
    (output_video_dir_tmp/"output.mp4").rename(video_output_fp)
    if remove_segments:
        shutil.rmtree(output_video_dir_tmp)
    return video_output_fp

def run_average_metrics(output_dir):
    output_dir = Path(output_dir)
    metadata_fps = list(output_dir.glob('**/*.metadata.json'))
    if len(metadata_fps) == 0:
        return
    category2wer = defaultdict(list)
    for idx, metadata_fp in enumerate(metadata_fps):
        with open(metadata_fp, 'r') as f:
            metadata = json.load(f)

        wer = metadata.get('wer', {})
        wer, ins, subs, dels = [wer.get(k, 0) for k in ['wer', 'ins', 'subs', 'dels']]
        mcs = metadata.get('mcs', 0)
        # update total metrics
        category_dir = metadata_fp.parent.resolve()
        if category_dir != output_dir.resolve(): # ignore category if there are none
            category2wer[str(category_dir)].append([wer, ins, subs, dels, mcs])
        category2wer[str(output_dir)].append([wer, ins, subs, dels, mcs]) # append to base directory to calculate total wer
        
    for dir_path, values in category2wer.items():
        metrics_fp = Path(dir_path)/'metrics.json'
        wer, ins, subs, dels, mcs = np.array(values).mean(axis=0)
        wer_metadata = {
            'wer': round(wer, 3),
            'ins': round(ins, 3),
            'subs': round(subs, 3),
            'dels': round(dels, 3),
        }
        update_json(metrics_fp, { 'wer': wer_metadata, 'mcs': round(mcs, 3) })


class AverageMetricsCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        with local_zero_first():
            if trainer.is_global_zero:
                try:
                    run_average_metrics(output_dir)
                except Exception as e:
                    print('Could not run average metrics:', e)

