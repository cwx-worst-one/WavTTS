import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio

from recipes.bigmusic.lightning.acoustic_modules import CoarseModule
from recipes.bigmusic.lightning.acoustic_modules import FineModule
from samantha.utils.hparams import DotDict
from recipes.musiclm.inference.utils import slugify, save_wav
from recipes.umm.modules.lit_module import Stage3
from transformers import BertTokenizer
import torch.functional
import importlib


SAMPLE_RATE = 24000

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission_batch):
        """Given a sequence emission over labels, get the best path
        Args:
          emission_batch (Tensor): Logit tensors. Shape `[batch, num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        res = []
        for emission in emission_batch:
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]
            joined = " ".join([self.labels[i] for i in indices])
            res.append(joined.replace("|", " ").strip())
        return res


def run_wer(umm_model, wavs, lyrics, verbose=True):
    vocab = list(umm_model.tokenizer.get_vocab().keys())
    greedy_decoder = GreedyCTCDecoder(vocab)
    wer_results = []
    input_batch = {"audio": wavs, "text": lyrics}
    model_input = umm_model.prepare_feature(input_batch)
    model_output = umm_model.model(model_input)
    emission, recon_feature = model_output["logits"], model_output["recon_feature"]
    actual_transcript = lyrics
    actual_transcript = [a.lower().replace(" <n> ", " ") for a in actual_transcript]
    greedy_transcript = greedy_decoder(emission)
    greedy_transcript = [g.lower().replace(" ' ", "'") for g in greedy_transcript]
    for j, (a, g) in enumerate(zip(actual_transcript, greedy_transcript)):
        greedy_wer = torchaudio.functional.edit_distance(a, g) / len(a)
        if verbose:
            print("=============================")
            print(f"Actual transcript: {a}")
            print(f"Greedy transcript: {g}")
            print(f"WER: {greedy_wer}")
        wer_results.append(greedy_wer)
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
        self.coarse_module = CoarseModule.load_from_checkpoint(self.extra_params.coarse_ckpt).eval()
        self.fine_module = FineModule.load_from_checkpoint(self.extra_params.fine_ckpt).eval()
        self.umm_module = Stage3.load_from_checkpoint(self.extra_params.umm_ckpt).eval()
        self.requires = {}
        self.wer_mcs = []
        self.load_required_modules()

    def load_required_modules(self):
        for name, item in self.hparams.required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))
        self.semantic_module.load_required_modules()
        self.umm_module.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    def _mcs(self, wavs, batch):
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
                self.semantic_module.requires, batch["style_audio"][:,0:10*SAMPLE_RATE], data_type='music'
            ).squeeze(1).to(self.device)
        audio_emb = self.semantic_module.input_embedders[mulan_emb_name].get_embeds(
            self.semantic_module.requires, wavs[:,0:10*SAMPLE_RATE], data_type='music'
        ).squeeze(1).to(self.device)
        mcs = torch.nn.functional.cosine_similarity(gt_emb, audio_emb).cpu().numpy()
        return mcs

    def run_metrics(self, wavs, batch):
        wer_results, actual_transcript, greedy_transcript = run_wer(self.umm_module, wavs.to(self.device), batch['lyrics'])
        mcs = self._mcs(wavs, batch)
        metrics = []
        for w, a, g, m in zip(wer_results, actual_transcript, greedy_transcript, mcs):
            metrics.append(
                f"MCS: {m}\nWER: {w}\nActual transcript: {a}\nGreedy transcript: {g}"
            )
        return torch.tensor(wer_results), mcs, metrics


    def _predict_step(self, batch, round):
        conditions = batch['conditions']                
        semantic_samples = self.semantic_module.predict(batch, self.extra_params)
        # gt_semantic_samples = self.semantic_module.target_embedder.tokenize(self.semantic_module.requires, batch['style_audio'][:, :24_000*10])
        coarse_samples = self.coarse_module.predict(semantic_samples, self.extra_params)
        fine_samples = self.fine_module.predict(coarse_samples, self.extra_params)
        bs = coarse_samples.size(0)

        coarse_samples = coarse_samples.view([bs, -1, self.extra_params.num_coarse])
        fine_samples = fine_samples.view([bs, -1, self.extra_params.num_fine])
        vqgan_inputs = (
            torch.cat([coarse_samples, fine_samples], dim=2)
            - torch.arange(self.extra_params.num_coarse + self.extra_params.num_fine, device=coarse_samples.device)
            * self.extra_params.soundstream_codebook_size
        )  # [b, t, n_codebook]
        vqgan_inputs = vqgan_inputs.transpose(
            1, 2
        )  # [b, t, n_codebook] -> [b, n_codebook, t]
        wavs = self.requires["ss_dec"](vqgan_inputs).squeeze(1)

        batch['generated_audio'] = wavs
        wer, mcs, metrics = self.run_metrics(wavs, batch)
        batch['metrics'] = metrics
        # batch['semantic_samples'] = semantic_samples
        # batch['gt_semantic_samples'] = gt_semantic_samples

        save_outputs(batch, round, self.extra_params.output_dir)
        # wavs_list = self.semantic_module.cal_eos(semantic_samples, wavs)
        return [wer.mean(), mcs.mean()]

    def predict_step(self, batch, batch_idx, dataloader_idx=0): 
        print("============== a new batch of size ", len(batch['lyrics']))       
        wer_mcs = []
        for i in range(self.extra_params.num_rounds):
            wer_mcs.append(self._predict_step(batch, i))
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

        self.umm_module = Stage3.load_from_checkpoint(self.extra_params.umm_ckpt).eval()
        self.coarse_module = CoarseModule.load_from_checkpoint(self.extra_params.coarse_ckpt).eval()
        self.fine_module = FineModule.load_from_checkpoint(self.extra_params.fine_ckpt).eval()
        self.requires = {}
        self.load_required_modules()
        self.wer = []

    def load_required_modules(self):
        for name, item in self.hparams.required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))
        self.coarse_module.load_required_modules()
        self.umm_module.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    def run_metrics(self, wavs, batch):
        if 'lyrics' not in batch: 
            return torch.zeros((wavs.shape[0])), None
        wer_results, actual_transcript, greedy_transcript = run_wer(self.umm_module, wavs, batch['lyrics'])
        metrics = []
        for w, a, g in zip(wer_results, actual_transcript, greedy_transcript):
            metrics.append(
                f"WER: {w}\nActual transcript: {a}\nGreedy transcript: {g}"
            )
        return torch.tensor(wer_results), metrics


    def _predict_step(self, batch, round):
        batch['target_audio'] = batch['style_audio'] # prepare_inputs expects target_audio key
        semantic_embeds = self.coarse_module.prepare_inputs_embeddings(batch)
        coarse_samples = self.coarse_module.predict(semantic_samples=None, hp=self.extra_params, semantic_embeds=semantic_embeds)
        fine_samples = self.fine_module.predict(coarse_samples, self.extra_params)
        bs = coarse_samples.size(0)
        
        coarse_samples = coarse_samples.view([bs, -1, self.extra_params.num_coarse])
        fine_samples = fine_samples.view([bs, -1, self.extra_params.num_fine])
        vqgan_inputs = (
            torch.cat([coarse_samples, fine_samples], dim=2)
            - torch.arange(self.extra_params.num_coarse + self.extra_params.num_fine, device=coarse_samples.device)
            * self.extra_params.soundstream_codebook_size
        )  # [b, t, n_codebook]
        vqgan_inputs = vqgan_inputs.transpose(
            1, 2
        )  # [b, t, n_codebook] -> [b, n_codebook, t]
        wavs = self.requires["ss_dec"](vqgan_inputs).squeeze(1)
        batch['generated_audio'] = wavs
        wer, metrics = self.run_metrics(wavs, batch)
        batch['metrics'] = metrics
        save_outputs(batch, round, self.extra_params.output_dir)
        return wer.mean()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):        
        wer = []
        for i in range(self.extra_params.num_rounds):
            wer.append(self._predict_step(batch, i))
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

def save_outputs(batch, round, output_dir):
    conditions = batch['conditions']
    lyrics = batch.get('lyrics')
    prompts = batch.get('style_text')
    categories = batch.get('category')
    style_audio = batch.get('style_audio')
    vocal_audio = batch.get('vocal_audio')
    metrics = batch.get('metrics')
    wavs = batch['generated_audio']
    semantic_samples = batch.get('semantic_samples')
    gt_semantic_samples = batch.get('gt_semantic_samples')
    for i, wav in enumerate(wavs):
        if categories is not None:
            wav_dir = os.path.join(output_dir, categories[i])
        else:
            wav_dir = output_dir
        os.makedirs(wav_dir, exist_ok=True)
        file_name = ""
        if 'lyrics_tokens' in conditions:
            file_name += slugify(lyrics[i])[:128]
        if 'style_text' in conditions:
            file_name += '--' + slugify(prompts[i])[:128]
        if file_name: 
            file_name += f'.{round}-{i}'
        else:
            file_name += f'{round}-{i}'
        wav_fp = os.path.join(wav_dir, f"{file_name}.wav")
        print(f"[Saving] {wav_fp}")
        save_wav(wav.cpu().float(), wav_fp, sr=SAMPLE_RATE)

        if semantic_samples is not None:
            semantic_fp = os.path.join(wav_dir, f"{file_name}.semantic.pt")
            torch.save(semantic_samples[i].cpu(), semantic_fp)
        if gt_semantic_samples is not None:
            semantic_fp = os.path.join(wav_dir, f"{file_name}.gt_semantic.pt")
            torch.save(gt_semantic_samples[i].cpu(), semantic_fp)

        txt_fp = os.path.join(wav_dir, f"{file_name}.txt")
        with open(txt_fp, 'w') as f:
            f.write(f'Conditions: {conditions}\n')
            if 'lyrics_tokens' in conditions:
                f.write(f'Lyrics: {lyrics[i]}\n')
            if 'style_text' in conditions:
                f.write(f'Prompt: {prompts[i]}\n')
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
            save_wav(style_audio[i].cpu().float(), input_wav_fp, sr=24000)

        if vocal_audio is not None:
            input_vocals_fp = os.path.join(wav_dir, f"{file_name}.vocal_prompt.wav")
            save_wav(vocal_audio[i].cpu().float(), input_vocals_fp, sr=24000)

