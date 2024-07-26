import pytorch_lightning as pl
import torch
from typing import Any, List, Union
import json
from pathlib import Path
import os
import textwrap
import shutil
import numpy as np
import glob
import tqdm
import librosa
import soundfile
from recipes.musiclm.inference.utils import slugify, save_wav, generate_hash, format_name, load_wav
from collections import defaultdict
from recipes.bigmusic.utils.format_utils import update_json
import numpy as np
from recipes.musiclm.utils.dist import local_zero_last, is_local_zero
from recipes.bigmusic.utils.upload import audio_tensor_to_bytes, upload_to_easycycle, upload_to_tos
from recipes.bigmusic.datasets.mir_data_util import ID_TEMPO_LABEL_MAP, ID_KEY_MAP
from recipes.bigmusic.datasets.utils.symbolic_music import pretty_midi_obj_to_midi_bytes


class SaveOutputsCallback(pl.Callback):
    def __init__(
        self,
        beam_size=1,
        samples_to_save=1,
        save_style_audio=True,
        save_mode="wav",
        save_semantic_tokens=False,
        save_mix_vocal_generated_audio=False,
        normalize_volume=False
    ):
        super().__init__()
        self.total_items = 0
        self.beam_size = beam_size
        self.samples_to_save = samples_to_save
        self.save_style_audio = save_style_audio
        self.save_mode = save_mode
        self.save_semantic_tokens = save_semantic_tokens
        self.save_mix_vocal_generated_audio = save_mix_vocal_generated_audio
        self.normalize_volume = normalize_volume

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
        if hasattr(pl_module, "semantic_module"):
            leadsheet_codec = pl_module.semantic_module.extra_params.get("leadsheet_codec")
        else: # GTInferenceModule does not have semantic_module
            leadsheet_codec = None
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
            save_mix_vocal_generated_audio=self.save_mix_vocal_generated_audio,
            save_mode=self.save_mode,
            normalize_volume=self.normalize_volume,
            save_semantic_tokens=self.save_semantic_tokens,
            leadsheet_codec=leadsheet_codec,
        )
        if isinstance(outputs['generated_audio_tensor'], list):
            outputs['generated_audio_tensor'] = outputs['generated_audio_tensor'][0]
        num_items = outputs['generated_audio_tensor'].shape[0] // self.beam_size
        self.total_items += num_items
        with open(Path(output_dir)/'inference_params.json', 'w', encoding='utf-8') as f:
            json.dump(pl_module.extra_params, f, indent=2)

        # save output paths so other callbacks can run metrics on audio
        if 'output_paths' in pl_module.extra_params:
            pl_module.extra_params['output_paths'].extend(output_paths)
        else:
            pl_module.extra_params['output_paths'] = output_paths

    def on_predict_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        if self.save_mode != "upload":
            return
        output_dir = pl_module.extra_params.output_dir
        with local_zero_last():
            if is_local_zero():
                output_dir = Path(output_dir)
                metadata_fps = list(output_dir.glob('**/*.metadata.json'))
                if len(metadata_fps) == 0:
                    return
                index_fname = os.path.join(output_dir, "index.csv")
                with open(index_fname, "w", encoding='utf-8') as fw:
                    fw.write("file_name,beam_id,audio_url\n")
                    for fp in metadata_fps:
                        with open(fp, "r", encoding='utf-8') as f:
                            metadata = json.load(f)
                        file_name = metadata["file_name"]
                        beam_id = metadata["index"]["beam_idx"]
                        audio_url = metadata["audio_url"]
                        fw.write(f"{file_name},{beam_id},{audio_url}\n")
                print(f"Wrote index to {index_fname}")
                summary_format = "default" if not self.save_mix_vocal_generated_audio else "singsong"
                summarize_uploaded_results(output_dir, format=summary_format)


def summarize_uploaded_results(output_dir, format="default"):
    if format not in ["default", "singsong"]:
        raise ValueError(f"Not supported format of summarying index csv: {format}")

    output_dir = Path(output_dir)
    metadata_fps = list(output_dir.glob('**/*.metadata.json'))
    if len(metadata_fps) == 0:
        return

    index_fname = os.path.join(output_dir, "index.csv")
    
    with open(index_fname, "w", encoding='utf-8') as fw:
        if format == "default":
            fw.write("file_name,beam_id,audio_url\n")
            for fp in metadata_fps:
                with open(fp, "r", encoding='utf-8') as f:
                    metadata = json.load(f)
                file_name = metadata["file_name"]
                beam_id = metadata["index"]["beam_idx"]
                audio_url = metadata["audio_url"]
                fw.write(f"{file_name},{beam_id},{audio_url}\n")  
        elif format == "singsong":
            fw.write("file_name,accomp_audio_url,vocal_audio_url,mixed_audio_url\n")
            for fp in metadata_fps:
                with open(fp, "r", encoding='utf-8') as f:
                    metadata = json.load(f)
                file_name = metadata["file_name"]
                accomp_audio_url = metadata["audio_url"]
                vocal_audio_url = metadata["vocal_audio_url"]
                mixed_audio_url = metadata["mixed_audio_url"]
                fw.write(f"{file_name},{accomp_audio_url},{vocal_audio_url},{mixed_audio_url}\n")
    print(f"Wrote index to {index_fname}")


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

def save_leadsheet(file_name, leadsheet_token, leadsheet_codec):
    try:
        midi_obj = leadsheet_codec.decode(leadsheet_token)
        midi_bytes = pretty_midi_obj_to_midi_bytes(midi_obj)
        print("Saving midi:", file_name)
        with open(file_name, "wb") as f:
            f.write(midi_bytes)
    except Exception as e:
        print("Decoding fail:", e)
    return

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
    save_mode="wav",
    normalize_volume=False,
    save_semantic_tokens=False,
    leadsheet_codec=None,
    save_mix_vocal_generated_audio=False,
):
    conditions = batch['conditions']
    if isinstance(conditions, list):
        conditions = conditions[0]
    index = batch.get('index')
    lyrics = batch.get('lyrics')
    lyrics_normalized_text = batch.get('lyrics_normalized_text')
    prompts = batch.get('original_style_text', batch.get('style_text'))
    style_categories = batch.get('style_category')
    structures = batch.get('structure')
    categories = batch.get('category')
    style_audio = batch.get('style_audio')
    vocal_audio = batch.get('vocal_audio')   
    target_audio = batch.get('target_audio')  
    metadatas = outputs.get('metadata')
    wavs = outputs['generated_audio']
    semantic_tokens = outputs.get('generated_semantic_tokens')
    leadsheet_tokens = outputs.get('generated_leadsheet_tokens')

    # Add these conditions (if in batch) to style_text and metadata.json
    cond_id_label_map = {
        'key': ID_KEY_MAP,
        'tempo_label': ID_TEMPO_LABEL_MAP
    }

    existing_extra_conds = {}
    for cond, id_label_map in cond_id_label_map.items():
        if cond not in batch:
            continue
        cond_labels = [id_label_map[int(x)] for x in batch[cond]]
        existing_extra_conds[cond] = cond_labels
        prompts = [f'{s},{o}' for s, o in zip(prompts, cond_labels)]  # add to style_text (prompts)

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
        # lyrics_str = lyrics[ii] if 'lyrics_tokens' in conditions else None
        lyrics_str = lyrics[ii] if ('lyrics_tokens' in conditions or 'leadsheet_tokens' in conditions ) else None
        lyrics_normalized_str = lyrics_normalized_text[ii] if 'lyrics_tokens' in conditions and lyrics_normalized_text else None        
        style_text = prompts[ii] if prompts else None
        style_category = style_categories[ii] if style_categories else None
        #style_category = style_categories[ii] if 'style_category' in conditions and style_categories else None
        structure = structures[ii] if 'structure' in conditions else None
        leadsheet_token = leadsheet_tokens[ii] if leadsheet_tokens is not None else None
        if index is None:
            file_name = f"{absolute_idx:03d}_{format_lyrics_and_style(style_text, lyrics_str)}"
        else:
            file_name = index[ii]
        wav_file_name = file_name
        if sample_round > 0:
            wav_file_name += ("_r" + str(prompt_idx % sample_round))
        if samples_to_save > 1 and beam_size > 1:
            wav_file_name += f".{beam_idx}"

        meta_fp = os.path.join(wav_dir, f"{wav_file_name}.metadata.json")
        metadata = metadatas[i] if metadatas is not None else {}

        # extra_conditions
        extra_cond_meta = {cond: cond_labels[i] for cond, cond_labels in existing_extra_conds.items()}

        metadata = {
            **metadata,
            'file_name': file_name,
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
                'csv_idx': index[ii] if index else absolute_idx,
                'beam_idx': beam_idx,
            },
            **extra_cond_meta,
        }
        
        wav_fp = os.path.join(wav_dir, f"{wav_file_name}.generated.wav")
        print(f"[Saving] {wav_fp}")
        save_mp3 = save_mode in ["mp3"]
        output_wav_fp = save_wav(wav.cpu().float(), wav_fp, sr=sample_rate, save_mp3=save_mp3, normalize_volume=normalize_volume)
        output_paths.append(output_wav_fp)

        if save_mode == "upload":
            try:
                audio_bytes = audio_tensor_to_bytes(wav.cpu().float(), sample_rate)
                metadata["audio_url"] = upload_to_easycycle(audio_bytes, f"{wav_file_name}.generated")
            except Exception as e:
                print('WARNING: Unable to upload file:', output_wav_fp, e)
                metadata["audio_url"] = "ERROR"

        if save_style_audio and style_audio is not None and beam_idx == 0:
            # style audio is always 24kHz (for now)
            if save_mode == "upload":
                audio_bytes = audio_tensor_to_bytes(style_audio[ii].cpu().float(), 24000)
                metadata["style_audio_url"] = upload_to_easycycle(audio_bytes, f"{file_name}.style_audio")
            else:
                input_wav_fp = os.path.join(wav_dir, f"{file_name}.style_audio.wav")
                save_wav(style_audio[ii].cpu().float(), input_wav_fp, sr=24000, save_mp3=save_mode == "mp3")

        if target_audio is not None and beam_idx == 0:
            # target audio is always 24kHz (for now)
            if save_mode == "upload":
                audio_bytes = audio_tensor_to_bytes(target_audio[ii].cpu().float(), 24000)
                metadata["target_audio_url"] = upload_to_easycycle(audio_bytes, f"{file_name}.target_audio")
            else:
                target_audio_fp = os.path.join(wav_dir, f"{file_name}.target_audio.wav")
                save_wav(target_audio[ii].cpu().float(), target_audio_fp, sr=24000, save_mp3=save_mode == "mp3")

        if vocal_audio is not None and beam_idx == 0:
            # vocal audio is always 24kHz (for now)
            if save_mode == "upload":
                audio_bytes = audio_tensor_to_bytes(vocal_audio[ii].cpu().float(), 24000)
                metadata["vocal_audio_url"] = upload_to_easycycle(audio_bytes, f"{file_name}.vocal_audio")
            else:
                input_vocals_fp = os.path.join(wav_dir, f"{file_name}.vocal_audio.wav")
                save_wav(vocal_audio[ii].cpu().float(), input_vocals_fp, sr=24000, save_mp3=save_mode == "mp3")
        
        if save_mix_vocal_generated_audio and vocal_audio is not None and beam_idx == 0:

            mixed_audio = mix_two_audio_tensors(vocal_audio[ii].cpu().float(), wav.cpu().float())

            if save_mode == "upload":
                audio_bytes = audio_tensor_to_bytes(mixed_audio, 24000)
                metadata["mixed_audio_url"] = upload_to_easycycle(audio_bytes, f"{file_name}.mixed_audio")
            else:
                input_vocals_fp = os.path.join(wav_dir, f"{file_name}.mixed_audio.wav")
                save_wav(mixed_audio, input_vocals_fp, sr=24000, save_mp3=save_mode == "mp3")
            

        if save_semantic_tokens and semantic_tokens is not None:
            semantic_tokens_fp = os.path.join(wav_dir, f"{wav_file_name}.semantic_tokens.pt")
            torch.save(semantic_tokens[i], semantic_tokens_fp)

        if leadsheet_token is not None and leadsheet_codec is not None:
            file_name = os.path.join(wav_dir, f"{wav_file_name}.mid")
            save_leadsheet(file_name, leadsheet_token, leadsheet_codec)

        print('Saving metadata', metadata)
        meta_fp = os.path.join(wav_dir, f"{wav_file_name}.metadata.json")
        with open(meta_fp, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    return output_paths


class SaveVideoCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        with local_zero_last():
            if is_local_zero():
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
            command = "ffmpeg-normalize '%s' -t -16 --keep-loudness-range-target -c:a libmp3lame -b:a 128k -o '%s' -f" % (generated_output_fp, generated_output_fp.replace(".wav", ".mp3"))
            os.system(command)
            os.remove(generated_output_fp)


def format_video_text(metadata, max_width=50):
    index = metadata['index']['csv_idx']
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

def save_video(input_results_dir, output_video_dir, format_video_text_fn=default_format_video_text, remove_segments=True, upload=True):
    colors = ["green", "blue", "brown"]
    output_video_dir = Path(output_video_dir)
    output_video_dir_tmp = output_video_dir/'tmp'
    output_video_dir_tmp.mkdir(exist_ok=True, parents=True)

    generated_output_fps = list(Path(input_results_dir).glob('**/*.generated.wav'))
    audio_format = ".wav"
    if len(generated_output_fps) == 0:
        generated_output_fps = list(Path(input_results_dir).glob('**/*.generated*.mp3'))
        audio_format = ".mp3"
    for idx, generated_output_fp in enumerate(generated_output_fps):
        audio_fp = generated_output_fp
        metadata_fp = str(generated_output_fp).replace('generated'+audio_format, 'metadata.json')
        
        output_video_fp = output_video_dir_tmp/generated_output_fp.with_suffix('.mp4').name
        output_text_fp = output_video_dir_tmp/generated_output_fp.with_suffix('.txt').name
        with open(metadata_fp, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        with open(output_text_fp, 'w', encoding='utf-8') as f:
            video_text, fontsize, line_spacing = format_video_text_fn(metadata)
            f.write(video_text)
        color = colors[idx % len(colors)]
        fontfile = "/usr/share/fonts/truetype/arphic/ukai.ttc"
        cmd = f'ffmpeg -y -v 0 -f lavfi -i color=c={color}:s=800x800:d=0.5 -i {audio_fp} -c:a aac -vf "drawtext=fontfile={fontfile}:fontsize={fontsize}:line_spacing={line_spacing}:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile={output_text_fp}" {output_video_fp} -v 0'
        os.system(cmd)

    # concat output videos
    video_output_fp = output_video_dir/f"{output_video_dir.name}.mp4"
    cmd_concat = f"cd {output_video_dir_tmp} && find *.mp4 | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt; ffmpeg -v 0 -f concat -i fl.txt -c copy output.mp4; rm fl.txt"
    os.system(cmd_concat)
    (output_video_dir_tmp/"output.mp4").rename(video_output_fp)
    if remove_segments:
        shutil.rmtree(output_video_dir_tmp)

    if upload and os.path.exists(video_output_fp):
        url = upload_to_tos(video_output_fp, "tmp/video_demo/")
        print("Saved video:", url)

        # update inference_params
        meta_fp = os.path.join(output_video_dir, "inference_params.json")
        with open(meta_fp, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        metadata["demo_video_url"] = url
        with open(meta_fp, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    return video_output_fp

def run_average_metrics(output_dir):
    output_dir = Path(output_dir)
    metadata_fps = list(output_dir.glob('**/*.metadata.json'))
    if len(metadata_fps) == 0:
        return
    category2wer = defaultdict(list)
    for idx, metadata_fp in enumerate(metadata_fps):
        with open(metadata_fp, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        wer = metadata.get('wer', {})
        wer, ins, subs, dels = [wer.get(k, 0) for k in ['wer', 'ins', 'subs', 'dels']]
        mcs = metadata.get('mcs', 0)
        chordprob = metadata.get('chordprob', -1)
        # update total metrics
        category_dir = metadata_fp.parent.resolve()
        if category_dir != output_dir.resolve(): # ignore category if there are none
            category2wer[str(category_dir)].append([wer, ins, subs, dels, mcs, chordprob])
        category2wer[str(output_dir)].append([wer, ins, subs, dels, mcs, chordprob]) # append to base directory to calculate total wer
        
    for dir_path, values in category2wer.items():
        metrics_fp = Path(dir_path)/'metrics.json'
        wer, ins, subs, dels, mcs, chordprob = np.array(values).mean(axis=0)
        wer_metadata = {
            'wer': round(wer, 3),
            'ins': round(ins, 3),
            'subs': round(subs, 3),
            'dels': round(dels, 3),
        }
        if chordprob >= 0:
            update_json(metrics_fp, { 'wer': wer_metadata, 'mcs': round(mcs, 3), 'chordprob': round(chordprob, 3) })
        else:
            update_json(metrics_fp, { 'wer': wer_metadata, 'mcs': round(mcs, 3) })


class AverageMetricsCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        with local_zero_last():
            if is_local_zero():
                try:
                    run_average_metrics(output_dir)
                except Exception as e:
                    print('Could not run average metrics:', e)


class MergeFullSongCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        infer_tag = pl_module.extra_params.infer_tag
        input_txt_dir = pl_module.extra_params.input_txt_pattern
        sample_rate = pl_module.extra_params.sample_rate
        predict_dir = pl_module.extra_params.output_dir
        merge_full_song(input_txt_dir, sample_rate, predict_dir, infer_tag)

def merge_full_song(input_txt_dir, sample_rate, predict_dir, infer_tag="", save_slice=False):
    print("Saving Fullsong:", input_txt_dir, sample_rate, predict_dir, infer_tag)
    demo_dir = os.path.join(predict_dir, "full_song_demo")
    sliced_demo = os.path.join(predict_dir, "sliced_demo")
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(sliced_demo, exist_ok=True)
    if infer_tag == "":
        infer_tag = os.path.basename(predict_dir)

    data_dict = {}
    for txt in glob.glob(f"{input_txt_dir}/*.txt"):
        txt_basename = os.path.basename(txt)
        wav_basename = txt_basename.replace(".txt", ".generated.wav")
        wav = os.path.join(predict_dir, wav_basename)
        slice_uttid = os.path.basename(txt).rstrip(".txt")
        slice_uttid = slice_uttid.replace("test_", "")
        song_uttid = slice_uttid[:-3]
        sliceindex = int(slice_uttid[-3:])

        if os.path.exists(wav):
            frame = open(txt).readlines()
            frame = np.sum([int(x.strip().split("\t")[2]) for x in frame])
            duration = frame * 0.0125
            nsample = int(duration * sample_rate)
            y, _ = librosa.load(wav, sr=sample_rate, mono=True, duration=duration)
            if len(y) < nsample:
                y = np.concatenate([y, np.zeros(nsample-len(y))], axis=0)
            # save_path = f"{sliced_demo}/{slice_uttid}_slice_{infer_tag}.wav"
            save_path = f"{sliced_demo}/0_{song_uttid}_{sliceindex}_{infer_tag}.wav"
            soundfile.write(save_path, y, sample_rate, "PCM_16")
        else:
            frame = int(open(txt).readlines()[0].split("\t")[2])
            nsample = int(sample_rate * frame * 0.0125)
            y = np.zeros([nsample])

        if song_uttid not in data_dict:
            data_dict[song_uttid] = []
        data_dict[song_uttid].append(y)

    for uttid, audio in data_dict.items():
        audio = np.concatenate(audio, axis=0)
        save_path = f"{demo_dir}/0_{uttid}_-1_{infer_tag}.wav"
        if len(audio) > 0:
            print("Generated Fullsong in:", save_path)
            soundfile.write(save_path, audio, sample_rate, "PCM_16")
        else:
            print("Empty Fullsong, please check input:", save_path, input_txt_dir, infer_tag, sample_rate, predict_dir)


def mix_two_audio_tensors(tensor_1, tensor_2):
    def to_stereo(audio):
        if audio.ndim == 1:
            return audio.unsqueeze(0).repeat((2,1))
        elif audio.ndim == 2:
            return audio
        else:
            raise ValueError(f"Cannot handle audio data with ndim: {audio.ndim}")

    tensor_1 = to_stereo(tensor_1)
    tensor_2 = to_stereo(tensor_2)

    if tensor_1.ndim != tensor_2.ndim:
        raise ValueError(f"Cannot mix tensors with different ndims: {tensor_1.ndim} and {tensor_2.ndim}")

    length = min(tensor_1.shape[-1], tensor_2.shape[-1])
    if tensor_1.ndim == 1:
        mixed = tensor_1[:length] + tensor_2[:length]
        return mixed / mixed.max()
    elif tensor_1.ndim == 2:
        mixed = tensor_1[:,:length] + tensor_2[:,:length]
        return mixed / mixed.max()
    
    raise ValueError(f"Cannot handle tensors with ndim as :{tensor_1.ndim}")
