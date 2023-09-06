import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio

from samantha.utils.hparams import DotDict
from recipes.musiclm.inference.utils import slugify, save_wav, generate_hash, format_name
from recipes.diffusion.models.diffusion_model.utils import run_diffusion
from recipes.bigmusic.lightning.embedding_modules import get_bestrq_umm_tokens
import torch.functional
import importlib
from recipes.bigmusic.utils.metrics_asr import wav2lyrics, edit_distance
from recipes.bigmusic.utils.model_initializer import run_2ar
from itertools import zip_longest

def run_wer(wavs, lyrics, verbose=True):
    wer_results = []
    _ = wav2lyrics(wavs)
    asr_lyrics, _ = wav2lyrics(wavs)
    actual_transcript = lyrics
    greedy_transcript = asr_lyrics
    actual_transcript = [a.lower().replace(" <n> ", " ").replace(',', '') for a in actual_transcript]
    for j, (a, g) in enumerate(zip(actual_transcript, greedy_transcript)):
        edits = edit_distance(a, g)
        # TODO: (AS) return detailed breakdown of ins, subs, dels
        wer = edits.edits() / len(a)
        if verbose:
            print("=============================")
            print(f"Actual transcript: {a}")
            print(f"Greedy transcript: {g}")
            print(f"WER: {wer}")
        wer_results.append(wer)
    return wer_results, actual_transcript, greedy_transcript

class SemanticInferenceModule(pl.LightningModule):
    def __init__(
        self,
        semantic_cls_path,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)

        *module_paths, cls_name = semantic_cls_path.split('.')
        module = importlib.import_module('.'.join(module_paths))
        semantic_class = getattr(module, cls_name)

        self.semantic_module = semantic_class.load_from_checkpoint(self.extra_params.semantic_ckpt).eval()
        self.requires = {}
        self.wer_mcs = []

        self.requires = {}
        
        required_modules = {}
        if self.extra_params.token2wav_type == 'diffusion':
            self.decoding_fn = run_diffusion
            required_modules.update(self.hparams.required_modules['diffusion_modules'])
            self.decoding_params = DotDict({ **self.extra_params, **extra_params['diffusion_params'] })
        elif self.extra_params.token2wav_type == 'ar':
            self.decoding_fn = run_2ar
            required_modules.update(self.hparams.required_modules['ar_modules'])
            self.decoding_params = DotDict({ **self.extra_params, **extra_params['ar_params'] })
        self.load_required_modules(required_modules)

    def load_required_modules(self, required_modules):
        for name, item in required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))
        self.semantic_module.load_required_modules()

    def _mcs(self, wavs, batch):
        sample_rate = self.extra_params.sample_rate
        mulan_max_duration = 10 * sample_rate
        if len(wavs.shape) == 3:
            wavs = wavs.squeeze(1)
        conditions = batch['conditions']
        mulan_emb_names = [key for key in self.semantic_module.input_embedders.keys() if key.startswith("mulan")]
        if not mulan_emb_names:
            return torch.zeros((wavs.shape[0]))
        mulan_emb_name = mulan_emb_names[0]
        # Compute MCS based on wavs
        if 'style_text' in conditions:
            gt_emb = self.semantic_module.input_embedders[mulan_emb_name].get_embeds(
                self.semantic_module.requires, batch["style_text"], data_type='text'
            ).squeeze(1).to(self.device)
        if 'style_audio' in conditions:
            gt_emb = self.semantic_module.input_embedders[mulan_emb_name].get_embeds(
                self.semantic_module.requires, batch["style_audio"][:,0:mulan_max_duration], data_type='music'
            ).squeeze(1).to(self.device)
        audio_emb = self.semantic_module.input_embedders[mulan_emb_name].get_embeds(
            self.semantic_module.requires, wavs[:,0:mulan_max_duration], data_type='music'
        ).squeeze(1).to(self.device)
        mcs = torch.nn.functional.cosine_similarity(gt_emb, audio_emb).cpu().numpy()
        return mcs

    def run_metrics(self, wavs, batch):
        wer_results, actual_transcript, greedy_transcript = run_wer(wavs.to(self.device), batch['lyrics'])
        mcs = self._mcs(wavs, batch)
        metrics = []
        for w, a, g, m in zip(wer_results, actual_transcript, greedy_transcript, mcs):
            metrics.append(
                f"MCS: {m}\nWER: {w}\nActual transcript: {a}\nGreedy transcript: {g}"
            )
        return torch.tensor(wer_results), mcs, metrics
    
    def process_eos_indexes(self, semantic_samples):
        semantic_frame_rate = self.semantic_module.extra_params.semantic_frame_rate
        sample_rate = self.extra_params.sample_rate
        bs = semantic_samples.shape[0]
        eos_id = self.semantic_module.target_embedder.eos_id
        eos_index_list = []
        if eos_id is not None:
            eos_padding_id = 0
            eos_mask = torch.cumsum(semantic_samples == eos_id, 1) > 0
            semantic_samples[eos_mask] = eos_padding_id
            token2wav_rate = int(sample_rate / semantic_frame_rate)
            eos_index_list = ((semantic_samples == eos_padding_id).bool().cumsum(axis=1) == 0).bool().sum(axis=1) * token2wav_rate
        return semantic_samples, eos_index_list


    def _predict_step(self, batch, round, batch_idx):
        semantic_samples = self.semantic_module.predict(batch, self.extra_params)
        semantic_samples, eos_index_list = self.process_eos_indexes(semantic_samples)
        wavs = self.decoding_fn(self.requires, semantic_samples, self.decoding_params)
        
        batch['generated_audio'] = wavs
        batch['eos_index_list'] = eos_index_list
        wer, mcs, metrics = self.run_metrics(wavs, batch)
        batch['metrics'] = metrics

        save_outputs(batch, round, batch_idx, self.extra_params.output_dir, self.extra_params.sample_rate)
        return [wer.mean(), mcs.mean()]

    def predict_step(self, batch, batch_idx, dataloader_idx=0): 
        print("============== a new batch of size ", len(batch['lyrics']))       
        wer_mcs = []
        for i in range(self.extra_params.num_rounds):
            wer_mcs.append(self._predict_step(batch, i, batch_idx))
        wer_mcs = np.array(wer_mcs).mean(axis=0)
        self.wer_mcs.append(wer_mcs)
        
    def on_predict_end(self):
        wer_mcs = np.array(self.wer_mcs).mean(axis=0)
        print("average metrics", wer_mcs)
        output_dir = self.extra_params.output_dir
        os.makedirs(output_dir, exist_ok=True)
        txt_fp = os.path.join(output_dir, "metrics.txt")
        with open(txt_fp, 'w') as f:
            f.write(f'avg wer: {wer_mcs[0]}\n')
            f.write(f'avg mcs: {wer_mcs[1]}\n')

class GTInferenceModule(pl.LightningModule):
    def __init__(
        self,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        
        required_modules = {}
        if self.extra_params.token2wav_type == 'diffusion':
            self.decoding_fn = run_diffusion
            required_modules.update(self.hparams.required_modules['diffusion_modules'])
            self.decoding_params = DotDict({ **self.extra_params, **extra_params['diffusion_params'] })
        elif self.extra_params.token2wav_type == 'ar':
            self.decoding_fn = run_2ar
            required_modules.update(self.hparams.required_modules['ar_modules'])
            self.decoding_params = DotDict({ **self.extra_params, **extra_params['ar_params'] })

        if self.extra_params.semantic_type == 'bestrq':
            required_modules.update(self.hparams.required_modules['bestrq_modules'])
            self.encoding_fn = get_bestrq_umm_tokens

        self.load_required_modules(required_modules)
        self.wer = []

    def load_required_modules(self, required_modules):
        for name, item in required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def run_metrics(self, wavs, batch):
        # TODO: (AS) put this in a callback
        if 'lyrics' not in batch: 
            return torch.zeros((wavs.shape[0])), None
        wer_results, actual_transcript, greedy_transcript = run_wer(wavs, batch['lyrics'])
        metrics = []
        for w, a, g in zip(wer_results, actual_transcript, greedy_transcript):
            metrics.append(
                f"WER: {w}\nActual transcript: {a}\nGreedy transcript: {g}"
            )
        return torch.tensor(wer_results), metrics


    def _predict_step(self, batch, round, batch_idx):
        batch['target_audio'] = batch['style_audio'] # prepare_inputs expects target_audio key
        semantic_samples = self.encoding_fn(self.requires, batch['target_audio'])
        wavs = self.decoding_fn(self.requires, semantic_samples, self.decoding_params) # 
        batch['generated_audio'] = wavs
        wer, metrics = self.run_metrics(wavs, batch)
        batch['metrics'] = metrics
        save_outputs(batch, round, batch_idx, self.extra_params.output_dir, self.extra_params.sample_rate)
        return wer.mean()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):        
        wer = []
        for i in range(self.extra_params.num_rounds):
            wer.append(self._predict_step(batch, i, batch_idx))
        wer = np.array(wer).mean(axis=0)
        self.wer.append(wer)
        print("batch metrics", wer)

    def on_predict_end(self):
        wer = np.array(self.wer).mean(axis=0)
        print("average metrics", wer)
        output_dir = self.extra_params.output_dir
        os.makedirs(output_dir, exist_ok=True)
        txt_fp = os.path.join(output_dir, "metrics.txt")
        with open(txt_fp, 'w') as f:
            f.write(f'avg wer: {wer}\n')

def format_lyrics_and_style(lyrics, style_text):
    lyrics_formated = slugify(lyrics) # this function was already there for MusicLM
    text_formated = slugify(style_text)
    text_combined = lyrics_formated + "_" + text_formated
    text_encoded = generate_hash(text_combined) # this function was already there for MusicLM
    name_formatted = lyrics_formated[:96] + "_" + text_formated[:32] + "_" + text_encoded[:4]
    return name_formatted

def save_outputs(batch, round, batch_idx, output_dir, sample_rate):
    conditions = batch['conditions']
    lyrics = batch.get('lyrics')
    prompts = batch.get('style_text')
    categories = batch.get('category')
    style_audio = batch.get('style_audio')
    vocal_audio = batch.get('vocal_audio')
    metrics = batch.get('metrics')
    wavs = batch['generated_audio']
    eos_index_list = batch.get('eos_index_list', [])
    for i, (eos, wav) in enumerate(zip_longest(eos_index_list, wavs)):
        if eos is not None:
            wav = wav[:eos]
        if categories is not None:
            wav_dir = os.path.join(output_dir, categories[i])
        else:
            wav_dir = output_dir
        os.makedirs(wav_dir, exist_ok=True)
        file_name = ""
        lyrics_str = lyrics[i] if 'lyrics_tokens' in conditions else None
        style_text = prompts[i] if 'style_text' in conditions else None
        if style_text and lyrics_str is None: # instrumental case. use old format
            file_name = f"{format_name(style_text)}.{round}-{i}-{batch_idx}"
        elif style_text and lyrics_str: # vocal case
            file_name = f"{format_lyrics_and_style(lyrics_str, style_text)}.{round}-{i}-{batch_idx}"
        else: # both style and lyrics are None, probably ground truth case
            file_name = f'{round}-{i}-{batch_idx}'
        wav_fp = os.path.join(wav_dir, f"{file_name}.wav")
        print(f"[Saving] {wav_fp}")
        save_wav(wav.cpu().float(), wav_fp, sr=sample_rate)

        txt_fp = os.path.join(wav_dir, f"{file_name}.txt")
        with open(txt_fp, 'w') as f:
            f.write(f'Conditions: {conditions}\n')
            if 'lyrics_tokens' in conditions:
                f.write(f'Lyrics: {lyrics_str}\n')
            if 'style_text' in conditions:
                f.write(f'Prompt: {style_text}\n')
            if metrics is not None:
                f.write(f'Metrics:\n {metrics[i]}\n')

        if 'lyrics_tokens' in conditions:
            txt_fp = os.path.join(wav_dir, f"{file_name}.lyrics.txt")
            l = lyrics[i]
            if l.endswith(" vocal"):
                l = l[:-len(" vocal")]
            with open(txt_fp, 'w') as f:
                f.write(l)
        if 'style_text' in conditions:
            txt_fp = os.path.join(wav_dir, f"{file_name}.prompt.txt")
            with open(txt_fp, 'w') as f:
                f.write(prompts[i])

        if style_audio is not None:
            input_wav_fp = os.path.join(wav_dir, f"{file_name}.audio_prompt.wav")
            save_wav(style_audio[i].cpu().float(), input_wav_fp, sr=sample_rate)

        if vocal_audio is not None:
            input_vocals_fp = os.path.join(wav_dir, f"{file_name}.vocal_prompt.wav")
            save_wav(vocal_audio[i].cpu().float(), input_vocals_fp, sr=sample_rate)

