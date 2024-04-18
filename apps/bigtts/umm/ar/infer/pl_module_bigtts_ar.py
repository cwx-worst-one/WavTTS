import logging
import os
from typing import Any

import librosa
import numpy as np
import torch
from pytorch_lightning import LightningModule

from samantha.utils.audio import load_wav
from samantha.utils.common import set_seed
from samantha.utils.cuda import torch_allow_tf32
from samantha.utils.hparams import DotDict

from ..data.lyrics import AddConditionsTransform, LyricsTokenTransform
from ..semantic_modules import SemanticModule_SpkidLLMV3_3, SemanticModule_SpkidLLMV3_5

logger = logging.getLogger(__name__)


def prepare_semantic_model(semantic_model_path, device, version):
    logger.info(f"Loading version: {version}")

    model_cls = {
        "spkid_v3_3": SemanticModule_SpkidLLMV3_3,
        "spkid_v3_5": SemanticModule_SpkidLLMV3_5,
    }
    if version not in model_cls:
        raise ValueError(
            f"Unknown semantic {version=}, valid versions are {list(model_cls.keys())}"
        )

    return (
        model_cls[version].load_from_checkpoint(semantic_model_path).to(device).eval()
    )


def prepare_umm(umm_ckpt_path, device):
    rank = int(device[-1])
    from recipes.umm.requires.model_initializer import init_stage3

    token_model = init_stage3(umm_ckpt_path, rank, "./")["Stage3"].eval()
    return token_model


class SemanticInference(LightningModule):
    def __init__(
        self,
        seed,
        semantic_precision,
        umm_ckpt_path,
        semantic_model_path,
        output_path,
        src_lang,
        tgt_lang,
        temperature=0.9,
        thresh=0.9,
        mode="naive",
        max_blank_length=5,
        step_out_blank: bool = False,
        version="spkid_v3_5",
        icl_mode="continuation",
    ):
        super().__init__()
        if semantic_precision == "bf16":
            self.semantic_precision = torch.bfloat16
        elif semantic_precision == "fp16":
            self.semantic_precision = torch.float16
        elif semantic_precision == "fp32":
            self.semantic_precision = torch.float32

        self.umm_ckpt_path = umm_ckpt_path
        self.output_path = output_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lang2id = {"en": 0, "zh": 1, "zh_en": 2}

        self.seed = seed
        set_seed(seed)
        self.semantic_model = prepare_semantic_model(
            semantic_model_path, self.device, version=version
        )
        self.icl_mode = icl_mode

        self.semantic_ar_model_hp = DotDict(
            {
                "duration": 60,
                "semantic_temperature": temperature,
                "semantic_thresh": thresh,
                "sample_mode": mode,
                "max_blank_length": max_blank_length,
                "step_out_blank": step_out_blank,
                "icl_mode": icl_mode,
            }
        )
        lyrics_max_seq_len = self.semantic_model.extra_params.get(
            "lyrics_max_seq_len", 600
        )
        self.lyrics_token_transform = LyricsTokenTransform.init_sami_tokenizer(
            lyrics_max_seq_len=lyrics_max_seq_len,
            truncate_long_lyrics=True,
            test_wer=True,
        )
        self.add_conditions_transform = AddConditionsTransform("lyrics_tokens")

    def preprocess_prompt_wav(self, prompt_wav_path, device):
        wav, sr = load_wav(prompt_wav_path, sr=24000)
        wav = torch.FloatTensor(wav).unsqueeze(0)
        scale = max(0.001, torch.max(torch.abs(wav)).item())
        wav = wav / scale * 0.95
        wav = wav.to(device)
        return wav

    def text2semantic(self, prompt_wav, prompt_lab, infer_lab):
        semantic_batch = {}
        prompt_umm_token = self.umm.wav2token(prompt_wav)
        # hard-code from @kainan.
        prompt_umm_token = prompt_umm_token[:, :-1]

        # # copy 2次，取前1/2
        # print(f"===> copy 3次，取前1/3")
        # prompt_umm_token = self.umm.wav2token(torch.cat([prompt_wav, prompt_wav, prompt_wav], -1))
        # prompt_umm_token = prompt_umm_token[:,:prompt_umm_token.size(-1) // 3]

        semantic_batch["audio_prompt"] = prompt_umm_token
        # prepare text-ids.
        ## 续写
        if self.icl_mode == "continuation":
            semantic_batch["lyrics"] = prompt_lab + "|" + infer_lab
        else:
            semantic_batch["lyrics"] = infer_lab  # 注意这里改了
        semantic_batch = self.lyrics_token_transform(semantic_batch)
        semantic_batch = self.add_conditions_transform(semantic_batch)
        # batching.
        for key in ["lyrics_tokens", "prompt_text_lens", "phones", "tones", "wordsegs"]:
            semantic_batch[key] = semantic_batch[key].unsqueeze(0)
        # language.
        semantic_batch["src_lang"] = torch.LongTensor([self.lang2id.get(self.src_lang)])
        semantic_batch["tgt_lang"] = torch.LongTensor([self.lang2id.get(self.tgt_lang)])

        with torch_allow_tf32(enable_matmul=True), torch.autocast(
            device_type="cuda", dtype=self.semantic_precision, enabled=True
        ):
            infer_umm_token = self.semantic_model.predict(
                semantic_batch, self.semantic_ar_model_hp
            )
        return infer_umm_token

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        set_seed(self.seed)
        uttid, prompt_lab, prompt_wav_path, infer_lab = batch
        prompt_wav = self.preprocess_prompt_wav(prompt_wav_path, self.device)

        infer_umm_token = self.text2semantic(prompt_wav, prompt_lab, infer_lab)
        if infer_umm_token is not None:
            infer_umm_token = infer_umm_token.detach().cpu().numpy()
            np.save(
                os.path.join(self.output_path, f"{uttid}.npy"),
                infer_umm_token,
                allow_pickle=False,
            )

    def setup(self, stage):
        device = f"cuda:{self.trainer.local_rank}"
        self.umm = prepare_umm(self.umm_ckpt_path, device)
