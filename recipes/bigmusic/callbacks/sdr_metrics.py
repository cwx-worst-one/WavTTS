import pytorch_lightning as pl
import torch
from recipes.musiclm.inference.utils import load_wav
import json
from pathlib import Path
import numpy as np
import tqdm
from recipes.bigmusic.utils.format_utils import normalize_text
from collections import defaultdict
import requests
import time
from uuid import uuid4
import base64
from recipes.bigmusic.callbacks.plot_metrics import plot_wer
from recipes.musiclm.utils.dist import local_zero_first

from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as sisdr
import librosa
from recipes.bigmusic.utils.format_utils import update_json

def load_wav(path, sr=24000):
    if path.endswith(".npy"):
        wav = np.load(path)
    elif path.endswith(".wav"):
        wav, sr = librosa.load(path, sr=sr, mono=False)
    else:
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(sr)
        wav = np.asarray(audio.get_array_of_samples())
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2_147_483_648.0
    
    if len(wav.shape) == 1:
        wav = torch.from_numpy(wav).unsqueeze(0).repeat(2,1)
    else:
        wav = torch.from_numpy(wav)
    # make sure wav shape is [2,T]
    return wav


class SDRCallback(pl.Callback):
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sdr = sdr
        self.sample_rate = sample_rate

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # process current chunk
        output_dir = pl_module.extra_params.output_dir
        (Path(output_dir)/f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS").touch()
        with local_zero_first(): 
            if trainer.is_global_zero:
                ts = time.time()
                while not all([(Path(output_dir)/f"{self.__class__.__name__}.{rank}.SUCCESS").exists() for rank in range(trainer.world_size)]):
                    time.sleep(10)
                    print(f"[{self.__class__.__name__}] waiting for all ranks done ... (cost {round(time.time() - ts, 3)}s)")                
                generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))

                run_sdr_metrics(generated_output_fps, sdr_func=self.sdr, sr=self.sample_rate)

                try:
                    output_sdr_for_all_samples_into_one_file(output_dir)
                except:
                    print(output_dir)
                    print ('failed to plot or generate sdr_metrics for all samples, for some unknown reason...')    # 只要文件夹结构没变，就不应该有问题，这里兜一下以防万一
            

        with open(Path(output_dir)/f'inference_params.json', 'w', encoding='utf-8') as f:
            json.dump(pl_module.extra_params, f, indent=2)


def run_sdr_metrics(generated_output_fps, sdr_func, sr=44100):
    def compute_sdr(gen_path, gt_path, sdr_func):
        
        gen_wav = load_wav(str(gen_path), sr)
        gt_wav = load_wav(str(gt_path), sr)
        if gen_wav.shape[-1] != gt_wav.shape[-1]:
            gen_wav = gen_wav[:, :min(gen_wav.shape[-1], gt_wav.shape[-1])]
            gt_wav = gt_wav[:, :min(gen_wav.shape[-1], gt_wav.shape[-1])]

        res = sdr_func(gen_wav, gt_wav)
        metadata = {
            'sdr_avg': float(res.mean()),
            'sdr_left': float(res[0]),
            'sdr_right': float(res[1]),
        }
        return metadata

    def merge_all_sdr(category2wer):
        all_ref_length = 0
        all_ins_err = 0
        all_subs_err = 0
        all_nums = 0
        wer_metadata_list = []
        for dir_path, sdrs in category2wer.items():
            metrics_fp = Path(dir_path)/'metrics.json'
            mean, left, right, nums = np.array(sdrs).sum(axis=0)
            metadata = {
                'sdr_avg': mean / nums, 
                'sdr_left': left / nums,
                'sdr_right': right / nums
            }
            all_ref_length += mean 
            all_ins_err += left 
            all_subs_err += right 
            all_nums += nums

            wer_metadata_list.append([metrics_fp, metadata])

        all_metadata = {
                'sdr_avg': all_ref_length / nums, 
                'sdr_left': all_ins_err / nums,
                'sdr_right': all_subs_err / nums
            }
        all_metrics_fp = Path(dir_path.rsplit('/', 1)[0])/'all_metrics.json'
        wer_metadata_list.append([all_metrics_fp, all_metadata])
        return wer_metadata_list

    category2wer = defaultdict(list)
    category2pinyinwer = defaultdict(list)

    for idx, generated_output_fp in enumerate(generated_output_fps):
        # asr_lyrics = run_asr_lyrics_sa_online(generated_output_fp, sdr_func)
        targetaudio_fp = str(generated_output_fp).replace('generated.wav', 'target_audio.wav')
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        
        # with open(metadata_fp, 'r', encoding='utf-8') as f:
        #     metadata = json.load(f)
        # # actual transcript

        sdr_metadata = {} 
        sdr_metadata['sdr'] = compute_sdr(generated_output_fp, targetaudio_fp, sdr_func)
        
        update_json(metadata_fp, sdr_metadata)
        if isinstance(generated_output_fp, str):
            import os
            category_dir = os.path.dirname(generated_output_fp)
        else:
            category_dir = generated_output_fp.parent.resolve()
        category2wer[str(category_dir)].append([
                sdr_metadata['sdr']['sdr_avg'],
                sdr_metadata['sdr']['sdr_left'],
                sdr_metadata['sdr']['sdr_right'],
                1
        ]) # append to base directory to calculate total wer

    for metrics_fp, wer_metadata in merge_all_sdr(category2wer):
        update_json(metrics_fp, {'sdr': wer_metadata})
        print(f"output_dir={metrics_fp}, SDR={wer_metadata}")


def output_sdr_for_all_samples_into_one_file(path_result):

    import os
    import glob
    import pandas as pd
    from tabulate import tabulate
    path_result = str(path_result)
    sdr = {}
    categories = [name for name in os.listdir(path_result) if os.path.isdir(os.path.join(path_result, name))]
    categories = [x for x in categories if x[0]!='.']
    sdr['all'] = {'sdr': [], 'sdr_left': [], 'sdr_right': []}

    title = ['index', 'sdr', 'sdr_left', 'sdr_right']
    df = pd.DataFrame(columns=title)

    for category in categories:
        sdr[category] = {'sdr': [], 'sdr_left': [], 'sdr_right': []}
        path_category = os.path.join(path_result, category)
        filenames = glob.glob(os.path.join(path_category, '*.metadata.json'))
        for filename in filenames:
            index = os.path.basename(filename).split('.')[0]
            metadata = json.load(open(filename, 'r', encoding='utf-8'))
            data = {}
            data['index'] = index
            data['sdr'] = metadata['sdr']['sdr_avg']
            data['sdr_left'] = metadata['sdr']['sdr_left']
            data['sdr_right'] = metadata['sdr']['sdr_right']
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    filename_out = os.path.join(path_result, 'sdr_for_all_samples.txt')
    table = tabulate(df, headers='keys', tablefmt='grid')
    print(table)
    with open(filename_out, 'w', encoding='utf-8') as f:
        f.write(table)

class SISDRCallback(pl.Callback):
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sisdr = sisdr
        self.sample_rate = sample_rate

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # process current chunk
        output_dir = pl_module.extra_params.output_dir
        (Path(output_dir)/f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS").touch()
        with local_zero_first(): 
            if trainer.is_global_zero:
                ts = time.time()
                while not all([(Path(output_dir)/f"{self.__class__.__name__}.{rank}.SUCCESS").exists() for rank in range(trainer.world_size)]):
                    time.sleep(10)
                    print(f"[{self.__class__.__name__}] waiting for all ranks done ... (cost {round(time.time() - ts, 3)}s)")                
                generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))

                run_sisdr_metrics(generated_output_fps, sisdr_func=self.sisdr, sr=self.sample_rate)

                try:
                    output_sisdr_for_all_samples_into_one_file(output_dir)
                except:
                    print(output_dir)
                    print ('failed to plot or generate sisdr_metrics for all samples, for some unknown reason...')    # 只要文件夹结构没变，就不应该有问题，这里兜一下以防万一
            

        with open(Path(output_dir)/f'inference_params.json', 'w', encoding='utf-8') as f:
            json.dump(pl_module.extra_params, f, indent=2)


def run_sisdr_metrics(generated_output_fps, sisdr_func, sr=44100):
    def compute_sisdr(gen_path, gt_path, sisdr_func):
        
        gen_wav = load_wav(str(gen_path), sr)
        gt_wav = load_wav(str(gt_path), sr)
        if gen_wav.shape[-1] != gt_wav.shape[-1]:
            gen_wav = gen_wav[:, :min(gen_wav.shape[-1], gt_wav.shape[-1])]
            gt_wav = gt_wav[:, :min(gen_wav.shape[-1], gt_wav.shape[-1])]

        res = sisdr_func(gen_wav, gt_wav)
        metadata = {
            'sisdr_avg': float(res.mean()),
            'sisdr_left': float(res[0]),
            'sisdr_right': float(res[1]),
        }
        return metadata

    def merge_all_sisdr(category2wer):
        all_ref_length = 0
        all_ins_err = 0
        all_subs_err = 0
        all_nums = 0
        wer_metadata_list = []
        for dir_path, sisdrs in category2wer.items():
            metrics_fp = Path(dir_path)/'metrics.json'
            mean, left, right, nums = np.array(sisdrs).sum(axis=0)
            metadata = {
                'sisdr_avg': mean / nums, 
                'sisdr_left': left / nums,
                'sisdr_right': right / nums
            }
            all_ref_length += mean 
            all_ins_err += left 
            all_subs_err += right 
            all_nums += nums

            wer_metadata_list.append([metrics_fp, metadata])

        all_metadata = {
                'sisdr_avg': all_ref_length / nums, 
                'sisdr_left': all_ins_err / nums,
                'sisdr_right': all_subs_err / nums
            }
        all_metrics_fp = Path(dir_path.rsplit('/', 1)[0])/'all_metrics.json'
        wer_metadata_list.append([all_metrics_fp, all_metadata])
        return wer_metadata_list

    category2wer = defaultdict(list)
    category2pinyinwer = defaultdict(list)

    for idx, generated_output_fp in enumerate(generated_output_fps):
        # asr_lyrics = run_asr_lyrics_sa_online(generated_output_fp, sisdr_func)
        targetaudio_fp = str(generated_output_fp).replace('generated.wav', 'target_audio.wav')
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        
        # with open(metadata_fp, 'r', encoding='utf-8') as f:
        #     metadata = json.load(f)
        # # actual transcript

        sisdr_metadata = {} 
        sisdr_metadata['sisdr'] = compute_sisdr(generated_output_fp, targetaudio_fp, sisdr_func)
        
        update_json(metadata_fp, sisdr_metadata)
        if isinstance(generated_output_fp, str):
            import os
            category_dir = os.path.dirname(generated_output_fp)
        else:
            category_dir = generated_output_fp.parent.resolve()
        category2wer[str(category_dir)].append([
                sisdr_metadata['sisdr']['sisdr_avg'],
                sisdr_metadata['sisdr']['sisdr_left'],
                sisdr_metadata['sisdr']['sisdr_right'],
                1
        ]) # append to base directory to calculate total wer

    for metrics_fp, wer_metadata in merge_all_sisdr(category2wer):
        update_json(metrics_fp, {'sisdr': wer_metadata})
        print(f"output_dir={metrics_fp}, SISDR={wer_metadata}")


def output_sisdr_for_all_samples_into_one_file(path_result):

    import os
    import glob
    import pandas as pd
    from tabulate import tabulate
    path_result = str(path_result)
    sisdr = {}
    categories = [name for name in os.listdir(path_result) if os.path.isdir(os.path.join(path_result, name))]
    categories = [x for x in categories if x[0]!='.']
    sisdr['all'] = {'sisdr': [], 'sisdr_left': [], 'sisdr_right': []}

    title = ['index', 'sisdr', 'sisdr_left', 'sisdr_right']
    df = pd.DataFrame(columns=title)

    for category in categories:
        sisdr[category] = {'sisdr': [], 'sisdr_left': [], 'sisdr_right': []}
        path_category = os.path.join(path_result, category)
        filenames = glob.glob(os.path.join(path_category, '*.metadata.json'))
        for filename in filenames:
            index = os.path.basename(filename).split('.')[0]
            metadata = json.load(open(filename, 'r', encoding='utf-8'))
            data = {}
            data['index'] = index
            data['sisdr'] = metadata['sisdr']['sisdr_avg']
            data['sisdr_left'] = metadata['sisdr']['sisdr_left']
            data['sisdr_right'] = metadata['sisdr']['sisdr_right']
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    filename_out = os.path.join(path_result, 'sisdr_for_all_samples.txt')
    table = tabulate(df, headers='keys', tablefmt='grid')
    print(table)
    with open(filename_out, 'w', encoding='utf-8') as f:
        f.write(table)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
    )
    # Add arguments for the two directory paths
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    args = parser.parse_args()
    s = SISDRCallback()
    # output_sisdr_for_all_samples_into_one_file(args.input_dir)
    run_sisdr_metrics(list(Path(args.input_dir).glob('**/*.generated.wav')), s.sisdr, 44100)