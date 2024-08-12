import math
import os
from functools import partial
from typing import Any
import random
import soundfile as sf
import scipy
from copy import deepcopy
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchaudio.functional import resample

from .infer_utils import load_torch_script, save_wav, set_seed
from .wvae import Wave
from samantha.models.speechdit import DualSampler



def compute_duration_score(text_id):
    phone_id = text_id[0]
    n_vowel_zh = torch.sum((phone_id >= 186) & (phone_id <= 262))
    n_vowel_en = torch.sum((phone_id >= 147) & (phone_id <= 164))
    n_symbol = torch.sum((phone_id >= 0) & (phone_id <= 122))
    # return n_vowel_zh + n_vowel_en * 0.8 + n_symbol * 1.2
    if n_vowel_zh > n_vowel_en:
        return n_vowel_zh + n_vowel_en * 1.3 + n_symbol
    else:
        return n_vowel_en * 1.3 + n_symbol


def compute_syn_duration(uttid, prompt_words_align, prompt_phones_align, prompt_phone_id, target_phone_id, prompt_dur):
    def get_sil_unit(wordseg_id, phone_id):
        word_seg = (wordseg_id == 3) | (wordseg_id == 5)
        en = ((phone_id>=123) & (phone_id<=164))
        zh = ((phone_id>=186) & (phone_id<=262)) 
        en_unit = torch.sum(wordseg_id & en)
        zh_unit = torch.sum(zh)
        return en_unit, zh_unit

    confidence = np.mean([x[3] for x in prompt_words_align])
    if confidence < 0.65:
        print(uttid, confidence, "alignment failed")
        prompt_duration_score = compute_duration_score(prompt_phone_id)
        syn_duration_score = compute_duration_score(target_phone_id)
        syn_dur = prompt_dur * syn_duration_score / prompt_duration_score
        if syn_dur + prompt_dur > 60:
            syn_dur = 60 - prompt_dur
        return syn_dur

    sum_word_dur = 0.0
    sum_sil_dur = 0.0
    for i, x in enumerate(prompt_phones_align):
        sum_word_dur += x[2] - x[1]
        if i > 0:
            sum_sil_dur += prompt_phones_align[i][1] - prompt_phones_align[i - 1][2]

    phone_id = prompt_phone_id[0][1:-1]
    wordseg_id = prompt_phone_id[2][1:-1]
    prompt_n_vowel_zh = torch.sum((phone_id>=186) & (phone_id<=262))
    prompt_n_vowel_en = torch.sum((phone_id>=147) & (phone_id<=164)) * 1.3
    prompt_n_symbol = torch.sum((phone_id>=0) & (phone_id<=122))
    prompt_n_unit = prompt_n_vowel_zh + prompt_n_vowel_en
    avg_unit_dur = sum_word_dur / prompt_n_unit

    prompt_n_sil_unit_en, prompt_n_sil_unit_zh = get_sil_unit(wordseg_id, phone_id)
    prompt_n_sil_unit = prompt_n_sil_unit_en + prompt_n_sil_unit_zh * 0.5
    avg_sil_dur = (sum_sil_dur - prompt_n_symbol * 0.5)/prompt_n_sil_unit
    avg_sil_dur = max(0, avg_sil_dur)
    avg_sil_dur = min(avg_sil_dur, 0.05)

    phone_id = target_phone_id[0][1:-1]
    wordseg_id = target_phone_id[2][1:-1]
    n_vowel_zh = torch.sum((phone_id>=186) & (phone_id<=262))
    n_vowel_en = torch.sum((phone_id>=147) & (phone_id<=164)) * 1.3
    n_unit = n_vowel_zh + n_vowel_en
    n_sil_unit_en, n_sil_unit_zh = get_sil_unit(wordseg_id, phone_id)
    n_sil_unit = n_sil_unit_en + n_sil_unit_zh * 0.5

    n_symbol = torch.sum((phone_id>=0) & (phone_id<=122))
    syn_dur = avg_unit_dur * n_unit + avg_sil_dur * n_sil_unit + n_symbol * 0.5 + 0.3
    if syn_dur + prompt_dur > 60:
        syn_dur = 60 - prompt_dur
    return syn_dur


def load_wav(fn, sr):
    wav, sample_rate = sf.read(fn)
    if len(wav.shape) > 1:
        wav = wav[:, 0]
    if sample_rate != sr:
        # wav = librosa.core.resample(wav, sample_rate, sr, res_type="kaiser_best")
        import scipy
        import scipy
        wav = scipy.signal.resample(wav, int(len(wav) * sr / sample_rate))
    return wav, sr


def prepare_diffusion_model(diffusion_ckpt_path, training_type, device):
    if training_type == "pretrain":
        from .lit_train import LitModule as pl_module
    elif training_type == "consistencydistill":
        from .lit_distill import ConsistencyDistill as pl_module
    elif training_type == "advdistill":
        from .lit_distill import AdvDistill as pl_module
    else:
        raise NotImplementedError

    model = pl_module.load_from_checkpoint(
        diffusion_ckpt_path, device=torch.device(device)
    ).model.to(device)
    model.eval()
    return model


def prepare_vocoder(vocoder_ckpt_path, device):
    vocoder = torch.jit.load(vocoder_ckpt_path, map_location=device).eval()
    return vocoder


class Norm:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def denorm(self, x):
        return (x * self.std) + self.mean

    def norm(self, x):
        return (x - self.mean) / self.std


class TTSInfer(LightningModule):
    def __init__(
        self,
        diffusion_ckpt_path,
        vocoder_ckpt_path,
        output_dir,
        training_type="pretrain",
        seed=1996,
        random_seed=False,
        save_prompt=False,
        diffusion_precision="bf16",
        diffusion_nfe=10,
        diffusion_sampler="ddim",
        cfg_w=1,
        cfg_text=False,
        norm_cfg=False,
        bn_config=None,
        use_phone_lang=False,
        length_factor=1,
        length_reference_path=None,
        use_adaptive_length=False,
        prompt_mode="pre",
        only_use_global_prompt=False,
        cla_prompt_mode=None,
        classifier_path=None,
        cg_w=1,
        classifier_label=1,
        speed_editing_mode=False,
    ):
        super().__init__()

        if random_seed:
            seed = set_seed(random.randint(0, 1000))
        else:
            seed = set_seed(seed)

        self.model = prepare_diffusion_model(diffusion_ckpt_path, training_type, self.device)
        if classifier_path is not None:
            from recipes.umm_classifier.modules.bn_classifier import BNClassifierModule as pl_module
            self.classifier = pl_module.load_from_checkpoint(classifier_path).model.to(self.device).eval()
        else:
            self.classifier = None
        self.cg_w = cg_w
        self.classifier_label = classifier_label

        self.bn_config = bn_config
        self.bn_norm = Norm(
            bn_config["bn_norm_mean"], bn_config["bn_norm_std"])

        self.vocoder_ckpt_path = vocoder_ckpt_path
        self.save_prompt = save_prompt
        self.diffusion_precision = diffusion_precision
        if self.diffusion_precision == "bf16":
            self.diffusion_precision = torch.bfloat16
        elif self.diffusion_precision == "fp16":
            self.diffusion_precision = torch.float16
        elif self.diffusion_precision == "fp32":
            self.diffusion_precision = torch.float32
        else:
            raise NotImplementedError
        self.diffusion_nfe = diffusion_nfe
        self.diffusion_sampler = diffusion_sampler
        self.cfg_w = cfg_w
        self.cfg_text = cfg_text
        self.norm_cfg = norm_cfg
        self.prompt_mode = prompt_mode
        self.use_phone_lang = use_phone_lang
        self.length_factor = length_factor
        self.length_reference_path = length_reference_path
        self.use_adaptive_length = use_adaptive_length
        self.only_use_global_prompt = only_use_global_prompt
        self.training_type = training_type
        self.speed_editing_mode = speed_editing_mode
        self.cla_prompt_mode = cla_prompt_mode
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def prepare_features(self, batch):

        if self.cla_prompt_mode == "cla-concat":
            print(batch)
            uttid, prompt_wav_path, prompt_text_id, syn_text_id, prompt_words_align, prompt_phones_align, raw_prompt_wav_path = batch
        # cla_prompt mode None or normal
        else:
            uttid, prompt_wav_path, prompt_text_id, syn_text_id, prompt_words_align, prompt_phones_align = batch

        if prompt_text_id is None or syn_text_id is None:
            return None

        device = f"cuda:{self.trainer.local_rank}"
        inputs = dict()

        # Prompt
        wav, _ = load_wav(prompt_wav_path, sr=24000)
        inputs["gt_wav"] = wav
        wav = torch.FloatTensor(wav).unsqueeze(0)
        scale = max(0.001, torch.max(torch.abs(wav)))
        wav = wav / scale * 0.95
        prompt_wav = wav.to(device)
        try:
            inputs["scale"] = scale.item()
        except:
            print("########")
            print(prompt_wav_path)
        # if self.prompt_mode == "post":
        #    prompt_wav = F.pad(prompt_wav, (2400, 0), "constant", 0)

        # Text
        if self.only_use_global_prompt:
            text_id = syn_text_id
        else:
            if self.prompt_mode == "pre":
                text_id = torch.cat([prompt_text_id, syn_text_id[:, 1:]], dim=-1)
            elif self.prompt_mode == "post":
                text_id = torch.cat([syn_text_id, prompt_text_id[:, 1:]], dim=-1)
            else:
                raise NotImplementedError
        text_id = F.pad(text_id, (0, 1), "constant", 1)
        inputs["frontend"] = {
            "phone": text_id[0:1, :].to(device),
            "tone": text_id[1:2, :].to(device),
            "word_seg": text_id[2:3, :].to(device),
        }
        if self.use_phone_lang:
            inputs["frontend"]["lang"] = text_id[3:4, :].to(device)

        prompt_bn = self.wvae.encode(prompt_wav)
        m, logs = torch.split(prompt_bn, 64, dim=-1)
        prompt_bn = m + torch.randn_like(m) * torch.exp(logs)
        prompt_bn = self.bn_norm.norm(prompt_bn)
        inputs["prompt_bn"] = prompt_bn.transpose(1, 2)  # [B,C,T]

        if self.cla_prompt_mode:
            # only for cla, concat prompt for extreme short scenarios
            if self.cla_prompt_mode == "cla-concat":
                raw_wav, _ = load_wav(raw_prompt_wav_path, sr=24000)
                raw_wav = torch.FloatTensor(raw_wav).unsqueeze(0)
                raw_scale = max(0.001, torch.max(torch.abs(raw_wav)))
                raw_wav = raw_wav / raw_scale * 0.95
                raw_prompt_wav = raw_wav.to(device)
                raw_prompt_bn = self.wvae.encode(raw_prompt_wav)
                raw_m, raw_logs = torch.split(raw_prompt_bn, 64, dim=-1)
                raw_prompt_bn = raw_m + torch.randn_like(raw_m) * torch.exp(raw_logs)
                raw_prompt_bn = self.bn_norm.norm(raw_prompt_bn)
                total_bn_len = int(prompt_bn.shape[1] + 1.1 * raw_prompt_bn.shape[1])
            else:
                total_bn_len = int(prompt_bn.shape[1] * (1 + 1.1))

        elif not self.speed_editing_mode:
            if self.use_adaptive_length:
                syn_dur = compute_syn_duration(uttid, prompt_words_align, prompt_phones_align, prompt_text_id, syn_text_id, prompt_bn.shape[1]/40)
                total_bn_len = prompt_bn.shape[1] + int(syn_dur * self.bn_config["hz"])
            else:
                if self.length_reference_path is None:
                    total_bn_len = int(prompt_bn.shape[1] * (self.length_factor * syn_text_id.shape[1]+prompt_text_id.shape[1]) / prompt_text_id.shape[1])
                else:
                    refer_wav_path = os.path.join(self.length_reference_path, uttid+".wav")
                    refer_wav, _ = load_wav(refer_wav_path, sr=24000)
                    total_bn_len = prompt_bn.shape[1] + int(len(refer_wav)//self.bn_config['hop_size'])
        else:
            total_bn_len = int(prompt_bn.shape[1] * (1 + 1/self.length_factor))
            print(total_bn_len, prompt_bn.shape[1])

        if self.only_use_global_prompt:
            inputs["bn_ctx"] = torch.ones([1, total_bn_len - prompt_bn.shape[1], prompt_bn.shape[2]], device=device) * \
                               self.bn_config['bn_padding']
            inputs["prompt_length"] = 0
        else:
            inputs["bn_ctx"] = torch.ones([1, total_bn_len, prompt_bn.shape[2]], device=device) * self.bn_config[
                'bn_padding']
            if self.prompt_mode == "pre":
                inputs["bn_ctx"][:, :inputs["prompt_bn"].shape[-1], :] = inputs["prompt_bn"].transpose(1, 2)
            elif self.prompt_mode == "post":
                inputs["bn_ctx"][:, -inputs["prompt_bn"].shape[-1]:, :] = inputs["prompt_bn"].transpose(1, 2)
            inputs["prompt_length"] = inputs["prompt_bn"].shape[-1]

        inputs["uttid"] = uttid

        if self.cfg_w != 1:
            inputs = self.make_cfg_input(inputs)

        return inputs

    def classifier_fn(self, bn, sigmas, y):
        with torch.enable_grad():
            bn_in = bn.detach().requires_grad_(True)
            logits = self.classifier.infer_with_noise(bn_in, sigmas)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[:, y].sum()
            grad = torch.autograd.grad(selected, bn_in)[0].float() * self.cg_w
        return grad

    def make_cfg_input(self, inputs):
        if self.cfg_w != 1:
            if self.cfg_text:
                inputs["frontend"]["phone"] = inputs["frontend"]["phone"].repeat(2, 1)
                inputs["frontend"]["phone"][1, :] = 1
                inputs["frontend"]["tone"] = inputs["frontend"]["tone"].repeat(2, 1)
                inputs["frontend"]["tone"][1, :] = 1
                inputs["frontend"]["word_seg"] = inputs["frontend"]["word_seg"].repeat(2, 1)
                inputs["frontend"]["word_seg"][1, :] = 1
                if "lang" in inputs["frontend"]:
                    inputs["frontend"]["lang"] = inputs["frontend"]["lang"].repeat(2, 1)
                    inputs["frontend"]["lang"][1, :] = 1
            else:
                inputs["frontend"]["phone"] = inputs["frontend"]["phone"].repeat(2, 1)
                inputs["frontend"]["tone"] = inputs["frontend"]["tone"].repeat(2, 1)
                inputs["frontend"]["word_seg"] = inputs["frontend"]["word_seg"].repeat(2, 1)
            inputs["bn_ctx"] = inputs["bn_ctx"].repeat(2, 1, 1)
            inputs["bn_ctx"][1, :] = self.bn_config["bn_padding"]
        return inputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs = self.prepare_features(batch)

        with torch.autocast(device_type="cuda", dtype=self.diffusion_precision, enabled=True):
            out = self.model.inference(inputs,
                                       self.diffusion_nfe,
                                       self.diffusion_sampler,
                                       cfg_w=self.cfg_w)
            # norm_cfg=self.norm_cfg,
            # prompt_mode=self.prompt_mode,
            # prompt_length=inputs["prompt_length"],
            # classifier_fn=partial(self.classifier_fn, y=self.classifier_label) if self.classifier is not None else None)
            # inpaint_x=inputs["prompt_bn"].transpose(1, 2))

        if self.prompt_mode == "pre" or inputs["prompt_length"] == 0:
            out = out[:, :, inputs["prompt_length"]:]
        elif self.prompt_mode == "post":
            out = out[:, :, :-inputs["prompt_length"]]

        z = out.float()
        z = self.bn_norm.denorm(z)
        output_wav = self.wvae.decoder_from_z(z.transpose(1, 2))

        audio = output_wav.squeeze().cpu().numpy()
        if "scale" not in inputs:
            inputs["scale"] = 1
        if "scale" not in inputs:
            inputs["scale"] = 1
        audio = audio * inputs["scale"] / 0.95
        audio = np.clip(audio, a_min=-1, a_max=1)

        if self.prompt_mode == "post":
            audio = audio[:-int(0.1 * 24000)]
        elif self.prompt_mode == "pre":
            audio = audio[int(0.1 * 24000):]

        if self.save_prompt:
            prompt_wav = inputs["gt_wav"]
            audio = np.concatenate([prompt_wav, np.ones([10]), audio])
        output_path = os.path.join(self.output_dir, inputs["uttid"] + ".wav")
        save_wav(audio, output_path)

    def wvae_reconstruct(self, batch):
        device = f"cuda:{self.trainer.local_rank}"
        uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id = batch
        # wav, _ = librosa.load(syn_wav_path, sr=24000, mono=True)
        wav, _ = load_wav(syn_wav_path, sr=24000)
        wav = torch.FloatTensor(wav).unsqueeze(0)
        scale = max(0.001, torch.max(torch.abs(wav)))
        wav = wav / scale * 0.95
        wav = wav.to(device)

        reconstruct_wav = self.wvae.reconstruct(wav)
        output_path = os.path.join(self.output_dir, uttid + ".wav")
        save_wav(reconstruct_wav.cpu().numpy(), output_path)

    def setup(self, stage):
        device = f"cuda:{self.trainer.local_rank}"
        self.wvae_encoder = load_torch_script(
            model_path=self.bn_config['wvae_encoder_path'],
            rank=self.trainer.local_rank,
            cache_dir=self.bn_config['wvae_cache_dir']
        )
        self.wvae_decoder = load_torch_script(
            model_path=self.bn_config['wvae_decoder_path'],
            rank=self.trainer.local_rank,
            cache_dir=self.bn_config['wvae_cache_dir']
        )
        self.wvae = Wave(self.wvae_encoder, self.wvae_decoder,
                         version=self.bn_config['wvae_version'],
                         hop_size=self.bn_config['wvae_encoder_hop_size'],
                         win_size=self.bn_config['wvae_encoder_win_size'])

    def get_bn(self, wav):
        bn = self.wvae.encode(wav)
        m, logs = torch.split(bn, 64, dim=-1)
        bn = m + torch.randn_like(m) * torch.exp(logs)
        bn = self.bn_norm.norm(bn)
        return bn

    def get_wav(self, wav_path, return_orig_wav=False):
        orig_wav, _ = load_wav(wav_path, sr=24000)
        wav = torch.FloatTensor(orig_wav).unsqueeze(0)
        scale = max(0.001, torch.max(torch.abs(wav)))
        wav = wav / scale * 0.95
        if return_orig_wav:
            return wav, scale, orig_wav
        else:
            return wav, scale

class TTSDualInfer(TTSInfer):
    def __init__(
        self,
        uncond_diffusion_ckpt_path,
        **kwargs):
        super().__init__(**kwargs)
        self.uncond_diffusion_ckpt_path = uncond_diffusion_ckpt_path
        self.uncond_model = prepare_diffusion_model(uncond_diffusion_ckpt_path, self.training_type, self.device)
        self.dual_sampler = DualSampler(self.model, self.uncond_model)

    def make_cfg_input(self, inputs):
        uncond_inputs = deepcopy(inputs)
        if self.cfg_text:
            uncond_inputs["frontend"]["phone"][:, :] = 1
            uncond_inputs["frontend"]["tone"][:, :] = 1
            uncond_inputs["frontend"]["word_seg"][:, :] = 1
            if "lang" in inputs["frontend"]:
                uncond_inputs["frontend"]["lang"][:, :] = 1
        uncond_inputs["bn_ctx"][:, :] = self.bn_config["bn_padding"]
        return inputs, uncond_inputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self.cfg_w != 1:
            inputs, uncond_inputs = self.prepare_features(batch)
        else:
            inputs = self.prepare_features(batch)
            uncond_inputs = inputs

        with torch.autocast(device_type="cuda", dtype=self.diffusion_precision, enabled=True): 
            out = self.dual_sampler.inference(inputs, uncond_inputs,
                    self.diffusion_nfe,
                    self.diffusion_sampler,
                    cfg_w=self.cfg_w)

        if self.prompt_mode == "pre" or inputs["prompt_length"] == 0:
            out = out[:, :, inputs["prompt_length"]:]
        elif self.prompt_mode == "post":
            out = out[:, :, :-inputs["prompt_length"]]

        z = out.float()
        z = self.bn_norm.denorm(z)
        output_wav = self.wvae.decoder_from_z(z.transpose(1, 2))

        audio = output_wav.squeeze().cpu().numpy()
        audio = audio * inputs["scale"] / 0.95
        audio = np.clip(audio, a_min=-1, a_max=1)

        if self.prompt_mode == "post":
            audio = audio[:-int(0.1*24000)]
        elif self.prompt_mode == "pre":
            audio = audio[int(0.1*24000):]

        if self.save_prompt:
            prompt_wav = inputs["gt_wav"]
            audio = np.concatenate([prompt_wav, np.ones([10]), audio])
        output_path = os.path.join(self.output_dir, inputs["uttid"]+".wav")
        save_wav(audio, output_path)
    
class EditFormalInfer(TTSInfer):
    def __init__(
        self,
        use_all_for_prompt=True,
        **kwargs):
        super().__init__(**kwargs)
        self.use_all_for_prompt = use_all_for_prompt

    def prepare_features(self, batch):
        device = f"cuda:{self.trainer.local_rank}"
        inputs = dict()

        uttid, orig_wav_path, acc_wav_path, syn_text_id, mask_info, confidence = batch
        if confidence < 0.6:
            print(f"{uttid}'s confidence is too low ({confidence})")
            return None

        #acc_wav, _ = self.get_wav(acc_wav_path, return_orig_wav=False)

        syn_wav, scale, orig_syn_wav = self.get_wav(orig_wav_path, return_orig_wav=True)
        syn_wav = syn_wav.to(device)
        inputs["scale"] = scale.item()

        # Text
        text_id = syn_text_id
        text_id = F.pad(text_id, (0, 1), "constant", 1)
        inputs["frontend"] = {
                "phone": text_id[0:1, :].to(device),
                "tone": text_id[1:2, :].to(device),
                "word_seg": text_id[2:3, :].to(device),
                }

        syn_bn = self.get_bn(syn_wav) # [B, T, C]

        inputs["bn_ctx"] = syn_bn.clone()
        inputs["bn_mask"] = torch.zeros([syn_bn.shape[1]])
        mask_change = 0
        for mask_start, mask_end, mask_add in mask_info:
            print(mask_start, mask_end, mask_add)
            #mask_add = mask_add.detach().cpu().numpy() # TODO: why cuda?
            mask_start = round(mask_start * self.bn_config["hz"]) + mask_change
            mask_end = round(mask_end * self.bn_config["hz"]) + mask_change
            mask_add = round(mask_add * self.bn_config["hz"])

            inputs["bn_ctx"][:, mask_start:mask_end] = self.bn_config["bn_padding"]
            inputs["bn_mask"][mask_start:mask_end] = 1
            if mask_add > 0:
                inputs["bn_ctx"] = torch.cat([
                    inputs["bn_ctx"][:, :mask_start],
                    torch.full((syn_bn.shape[0], mask_add, syn_bn.shape[-1]), self.bn_config["bn_padding"], device=device),
                    inputs["bn_ctx"][:, mask_start:],
                    ], dim=1)
                inputs["bn_mask"] = torch.cat([
                    inputs["bn_mask"][:mask_start],
                    torch.ones((mask_add)),
                    inputs["bn_mask"][mask_start:],
                    ])

            elif mask_add < 0:
                inputs["bn_ctx"] = torch.cat([
                    inputs["bn_ctx"][:, :mask_start],
                    inputs["bn_ctx"][:, mask_start-mask_add:],
                    ], dim=1)
                inputs["bn_mask"] = torch.cat([
                    inputs["bn_mask"][:mask_start],
                    inputs["bn_mask"][mask_start-mask_add:],
                    ])

            mask_change += mask_add

        if self.use_all_for_prompt:
            inputs["prompt_bn"] = syn_bn.transpose(1, 2) # [B,C,T]
        else:
            raise NotImplementedError

        inputs["uttid"] = uttid
        #inputs["acc_wav"] = acc_wav

        if self.cfg_w != 1:
            inputs = self.make_cfg_input(inputs)
        return inputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs = self.prepare_features(batch)
        if inputs is None:
            return 
        unmask_idx = torch.where(inputs["bn_mask"]==0)[0]

        with torch.autocast(device_type="cuda", dtype=self.diffusion_precision, enabled=True): 
            out = self.model.inference(inputs, 
                    self.diffusion_nfe,
                    self.diffusion_sampler,
                    cfg_w=self.cfg_w).transpose(1, 2)

        out = out.float().detach()
        out[:, unmask_idx] = inputs["bn_ctx"][0, unmask_idx]
        z = self.bn_norm.denorm(out)
        output_wav = self.wvae.decoder_from_z(z)

        audio = output_wav.squeeze().cpu().numpy()
        audio = audio * inputs["scale"] / 0.95
        audio = np.clip(audio, a_min=-1, a_max=1)
        output_path = os.path.join(self.output_dir, inputs["uttid"]+".wav")
        save_wav(audio, output_path)
      

        
class EditInfer(TTSInfer):
    def __init__(
            self,
            mask_part=0.4,
            use_all_for_prompt=True,
            **kwargs):
        super().__init__(**kwargs)
        self.mask_part = mask_part
        self.use_all_for_prompt = use_all_for_prompt

    def prepare_features(self, batch):
        uttid, syn_wav_path, syn_text_id = batch

        device = f"cuda:{self.trainer.local_rank}"
        inputs = dict()

        # Prompt
        syn_wav, scale, orig_syn_wav = self.get_wav(syn_wav_path, return_orig_wav=True)
        syn_wav = syn_wav.to(device)
        inputs["scale"] = scale.item()

        # Text
        text_id = syn_text_id
        text_id = F.pad(text_id, (0, 1), "constant", 1)
        inputs["frontend"] = {
            "phone": text_id[0:1, :].to(device),
            "tone": text_id[1:2, :].to(device),
            "word_seg": text_id[2:3, :].to(device),
        }

        syn_bn = self.get_bn(syn_wav)  # [B, T, C]

        inputs["bn_ctx"] = syn_bn.clone()
        mask_len = int(syn_bn.shape[1] * self.mask_part)
        mask_start = int(torch.rand([1]) * (syn_bn.shape[1] - mask_len))
        inputs["bn_ctx"][:, mask_start:mask_start + mask_len] = self.bn_config["bn_padding"]
        if self.use_all_for_prompt:
            inputs["prompt_bn"] = syn_bn.transpose(1, 2)  # [B,C,T]
        else:
            if mask_start + mask_len < syn_bn.shape[1]:
                inputs["prompt_bn"] = torch.cat([syn_bn[:, :mask_start], syn_bn[:, :mask_start + mask_len]],
                                                dim=1).transpose(1, 2)
            else:
                inputs["prompt_bn"] = syn_bn[:, :mask_start].transpose(1, 2)

        inputs["uttid"] = uttid
        inputs["bn_mask"] = torch.zeros([syn_bn.shape[1]])
        inputs["bn_mask"][mask_start:mask_start + mask_len] = 1

        if self.cfg_w != 1:
            inputs = self.make_cfg_input(inputs)
        return inputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs = self.prepare_features(batch)
        unmask_idx = torch.where(inputs["bn_mask"] == 0)[0]

        with torch.autocast(device_type="cuda", dtype=self.diffusion_precision, enabled=True):
            out = self.model.inference(inputs,
                                       self.diffusion_nfe,
                                       self.diffusion_sampler,
                                       cfg_w=self.cfg_w).transpose(1, 2)

        out = out.float().detach()
        out[:, unmask_idx] = inputs["bn_ctx"][0, unmask_idx]
        z = self.bn_norm.denorm(out)
        output_wav = self.wvae.decoder_from_z(z)

        audio = output_wav.squeeze().cpu().numpy()
        audio = audio * inputs["scale"] / 0.95
        audio = np.clip(audio, a_min=-1, a_max=1)
        output_path = os.path.join(self.output_dir, inputs["uttid"] + ".wav")
        save_wav(audio, output_path)


class VCInfer(TTSInfer):
    def __init__(
            self,
            backward_t=1,
            **kwargs):
        super().__init__(**kwargs)
        self.backward_t = backward_t

    def prepare_features(self, batch):
        uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id = batch

        if prompt_text_id is None or syn_text_id is None:
            return None

        device = f"cuda:{self.trainer.local_rank}"
        inputs = dict()

        # Prompt
        prompt_wav, scale, orig_prompt_wav = self.get_wav(prompt_wav_path, return_orig_wav=True)
        prompt_wav = prompt_wav.to(device)
        inputs["scale"] = scale.item()
        inputs["gt_wav"] = orig_prompt_wav

        syn_wav, _ = self.get_wav(syn_wav_path)
        syn_wav = syn_wav.to(device)

        # Text
        if self.prompt_mode == "pre":
            text_id = torch.cat([prompt_text_id, syn_text_id[:, 1:]], dim=-1)
        elif self.prompt_mode == "post":
            text_id = torch.cat([syn_text_id, prompt_text_id[:, 1:]], dim=-1)
        else:
            raise NotImplementedError
        text_id = F.pad(text_id, (0, 1), "constant", 1)
        inputs["frontend"] = {
            "phone": text_id[0:1, :].to(device),
            "tone": text_id[1:2, :].to(device),
            "word_seg": text_id[2:3, :].to(device),
        }
        if self.use_phone_lang:
            inputs["frontend"]["lang"] = text_id[3:4, :].to(device)

        prompt_bn = self.get_bn(prompt_wav)
        inputs["prompt_bn"] = prompt_bn.transpose(1, 2)  # [B,C,T]

        syn_bn = self.get_bn(syn_wav)

        total_bn_len = prompt_bn.shape[1] + syn_bn.shape[1]

        inputs["bn_ctx"] = torch.ones([1, total_bn_len, prompt_bn.shape[2]], device=device) * self.bn_config[
            'bn_padding']
        if self.prompt_mode == "pre":
            inputs["bn_ctx"][:, :inputs["prompt_bn"].shape[-1], :] = inputs["prompt_bn"].transpose(1, 2)
            inputs["syn_bn"] = torch.cat([prompt_bn, syn_bn], dim=1)
        elif self.prompt_mode == "post":
            inputs["bn_ctx"][:, -inputs["prompt_bn"].shape[-1]:, :] = inputs["prompt_bn"].transpose(1, 2)
            inputs["syn_bn"] = torch.cat([syn_bn, prompt_bn], dim=1)
        inputs["prompt_length"] = inputs["prompt_bn"].shape[-1]

        inputs["uttid"] = uttid

        if self.cfg_w != 1:
            inputs = self.make_cfg_input(inputs)

        return inputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs = self.prepare_features(batch)

        if self.diffusion_precision == "bf16":
            dtype = torch.bfloat16
        elif self.diffusion_precision == "fp16":
            dtype = torch.float16
        elif self.diffusion_precision == "fp32":
            dtype = torch.float32
        else:
            raise NotImplementedError
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out = self.model.inference(inputs,
                                       self.diffusion_nfe,
                                       self.diffusion_sampler,
                                       cfg_w=self.cfg_w,
                                       norm_cfg=self.norm_cfg,
                                       syn_bn=inputs["syn_bn"],
                                       backward_t=self.backward_t)

        if self.prompt_mode == "pre":
            out = out[:, :, inputs["prompt_length"]:]
        elif self.prompt_mode == "post":
            out = out[:, :, :-inputs["prompt_length"]]

        out = out.float()
        z = out
        z = self.bn_norm.denorm(z)
        output_wav = self.wvae.decoder_from_z(z.transpose(1, 2))

        audio = output_wav.squeeze().cpu().numpy()
        audio = audio * inputs["scale"] / 0.95
        audio = np.clip(audio, a_min=-1, a_max=1)

        if self.save_prompt:
            prompt_wav = inputs["gt_wav"]
            audio = np.concatenate([prompt_wav, np.ones([10]), audio])
        output_path = os.path.join(self.output_dir, inputs["uttid"] + ".wav")
        save_wav(audio, output_path)
