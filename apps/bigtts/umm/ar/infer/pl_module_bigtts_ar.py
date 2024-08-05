import logging
import os
from typing import Any

import numpy as np
import torch

from pytorch_lightning import LightningModule

from samantha.utils.audio import load_wav
from samantha.utils.common import set_seed
from samantha.utils.cuda import torch_allow_tf32
from samantha.utils.hparams import DotDict
from samantha.dataio.remote_io import load_json

from transformers import LlamaTokenizer, T5Tokenizer, AutoTokenizer

from ..data.lyrics import AddConditionsTransform, LyricsTokenTransform
from apps.bigtts.umm.ar.lightning import (
    SemanticModule_Valle,
    SemanticModule_MergeV2,
    SemanticModule_MergeV2_1,
)
from apps.bigtts.umm.ar.data.utils import get_text_lang_ids
from ..utility.umm_initializer import init_umm

logger = logging.getLogger(__name__)


def prepare_semantic_model(semantic_model_path, device, version):
    logger.info(f"Loading version: {version}")

    model_cls = {
        "merge_v1": SemanticModule_Valle,
        "merge_v2": SemanticModule_MergeV2,
        "merge_v2_1": SemanticModule_MergeV2_1,
    }
    if version not in model_cls:
        raise ValueError(
            f"Unknown semantic {version=}, valid versions are {list(model_cls.keys())}"
        )

    return (
        model_cls[version]
        .load_from_checkpoint(semantic_model_path, map_location=device)
        .eval()
    )


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
        max_repeat_times=1,
        step_out_blank: bool = False,
        version="merge_v2",
        icl_mode="continuation",
        eos_weight=1.0,
        cfg_config=[False, 0.0],
        umm_version="0.6.2",
        use_spk_id: bool = False,
        spk2id: str = None,
        speaker_name: str = None,
        use_spk_tag: bool = False,
        tag_id: int = 0,
        use_bpe: bool = False,
        bpe_type: str = "llama",
        bpe_dir: str = "",
        use_text_lang_embedding: bool = False,
        textlang2id: str = "",
        use_pre_utt: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        if semantic_precision == "bf16":
            self.semantic_precision = torch.bfloat16
        elif semantic_precision == "fp16":
            self.semantic_precision = torch.float16
        elif semantic_precision == "fp32":
            self.semantic_precision = torch.float32

        self.use_spk_id = use_spk_id
        if self.use_spk_id:
            self.spk2id = load_json(spk2id)
            self.speaker_name = speaker_name

        self.use_text_lang_embedding = use_text_lang_embedding
        if self.use_text_lang_embedding:
            self.textlang2id = load_json(textlang2id)
            print(f"{self.textlang2id=}")

        self.use_bpe = use_bpe
        self.bpe_type = bpe_type
        self.bpe_dir = bpe_dir

        if self.use_bpe:
            print(f"##### Using BPE {self.bpe_type} #####")
            if self.bpe_type == "flan-T5-large":
                self.bpe_tokenizer = T5Tokenizer.from_pretrained(bpe_dir)
            elif self.bpe_type == "byte-T5-base":
                self.bpe_tokenizer = AutoTokenizer.from_pretrained(bpe_dir)
            elif self.bpe_type == "llama":
                self.bpe_tokenizer = LlamaTokenizer.from_pretrained(bpe_dir)
            else:
                raise NotImplementedError
        else:
            self.bpe_tokenizer = None

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lang2id = {"en": 0, "zh": 1, "zh_en": 2}

        self.use_spk_tag = use_spk_tag
        self.tag_id = tag_id

        set_seed(seed)
        self.eos_weight = float(eos_weight)

        self.semantic_ar_model_hp = DotDict(
            {
                "duration": 60,
                "semantic_temperature": temperature,
                "semantic_thresh": thresh,
                "sample_mode": mode,
                "max_blank_length": max_blank_length,
                "max_repeat_times": max_repeat_times,
                "step_out_blank": step_out_blank,
                "icl_mode": icl_mode,
                "cfg_config": cfg_config,
                "eos_weight": eos_weight,
            }
        )

    def setup(self, stage):
        self.semantic_model = prepare_semantic_model(
            self.hparams.semantic_model_path, self.device, version=self.hparams.version
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
        self.umm = init_umm(
            self.hparams.umm_ckpt_path, self.device, self.hparams.umm_version
        )

    def preprocess_prompt_wav(self, prompt_wav_path, device, use_pad=True):
        pad_width = (960 * 1, 960 * 2) if use_pad else (0, 0)
        wav, _ = load_wav(prompt_wav_path, sr=24000, pad_width=pad_width)
        wav = torch.FloatTensor(wav).unsqueeze(0)
        scale = max(0.001, torch.max(torch.abs(wav)).item())
        wav = wav / scale * 0.95
        wav = wav.to(device)
        return wav

    def convert_lyrics_to_labels(self, lyrics):
        if "|" not in lyrics:  # no prompt
            labels = lyrics.split("\n")
            return labels

        prompt_lab, infer_lab = lyrics.split("|")
        prompt_labels = prompt_lab.split("\n")
        infer_labels = infer_lab.split("\n")
        labels = prompt_labels + infer_labels
        return labels

    def text2semantic_v1(
        self, prompt_wav, prompt_lab, prompt_text, infer_lab, infer_text, pre_utt
    ):
        semantic_batch = {}

        if self.use_bpe:
            if pre_utt["text"] is not None:
                bpe_seq = np.asarray(
                    self.bpe_tokenizer(
                        pre_utt["text"] + infer_text, truncation=True, max_length=2048
                    ).input_ids
                )
                semantic_batch["bpes"] = torch.LongTensor(bpe_seq).unsqueeze(0)
            else:
                bpe_seq = np.asarray(
                    self.bpe_tokenizer(
                        infer_text, truncation=True, max_length=2048
                    ).input_ids
                )
                semantic_batch["bpes"] = torch.LongTensor(bpe_seq).unsqueeze(0)

        cur_uttr_lang = self.get_lab_lang(infer_lab)
        if pre_utt["token"] is not None:
            prompt_umm_token = pre_utt["token"]
            prompt_umm_token = prompt_umm_token[:, :-2]  # hard-code from @kainan.
        else:
            prompt_umm_token = None
        if (pre_utt["lang"] is None) or (pre_utt["lang"] != cur_uttr_lang):
            prompt_umm_token = None
        semantic_batch["audio_prompt"] = prompt_umm_token

        if self.use_spk_id:
            semantic_batch["speaker_id"] = torch.LongTensor(
                [self.spk2id.get(self.speaker_name) + 1]
            )
        if self.use_spk_tag:
            semantic_batch["tag_id"] = torch.LongTensor([self.tag_id + 1])

        # prepare text-ids.
        if (pre_utt["lang"] is not None) and (pre_utt["lang"] == cur_uttr_lang):
            prompt_lab = pre_utt["lab"]
            semantic_batch["lyrics"] = prompt_lab + "|" + infer_lab
        else:
            semantic_batch["lyrics"] = infer_lab
        semantic_batch = self.lyrics_token_transform(semantic_batch)
        semantic_batch = self.add_conditions_transform(semantic_batch)

        # batching.
        for key in [
            "lyrics_tokens",
            "prompt_text_lens",
            "phones",
            "tones",
            "wordsegs",
            # cfg config
            "phones_cfg",
            "tones_cfg",
            "wordsegs_cfg",
        ]:
            semantic_batch[key] = semantic_batch[key].unsqueeze(0)

        # language.
        semantic_batch["src_lang"] = torch.LongTensor(
            [self.lang2id.get(self.hparams.src_lang)]
        )
        semantic_batch["tgt_lang"] = torch.LongTensor(
            [self.lang2id.get(self.hparams.tgt_lang)]
        )

        if self.use_text_lang_embedding:
            labels = self.convert_lyrics_to_labels(semantic_batch["lyrics"])
            # Get text language ID
            text_lang_ids, _ = get_text_lang_ids(labels, self.textlang2id)
            text_lang_ids += 1
            semantic_batch["text_lang"] = (
                torch.from_numpy(text_lang_ids).long().to(self.device)
            )

        with (
            torch_allow_tf32(enable_matmul=False),
            torch.autocast(
                device_type="cuda", dtype=self.semantic_precision, enabled=True
            ),
        ):
            infer_umm_token = self.semantic_model.predict(
                semantic_batch, self.semantic_ar_model_hp
            )
        return infer_umm_token

    def text2semantic_v2(
        self, prompt_wav, prompt_lab, prompt_text, infer_lab, infer_text, pre_utt
    ):
        semantic_batch = {}

        # prepare umm token
        # copy 3次，取前1/3
        prompt_umm_token = self.umm.wav2token(
            torch.cat([prompt_wav, prompt_wav, prompt_wav], -1)
        )
        prompt_umm_token = prompt_umm_token[:, : prompt_umm_token.size(-1) // 3]
        if pre_utt["token"] is not None:
            prompt_umm_token = torch.concatenate(
                [prompt_umm_token, pre_utt["token"]], dim=1
            )

        # hard-code from @kainan.
        prompt_umm_token = prompt_umm_token[:, :-1]
        semantic_batch["audio_prompt"] = prompt_umm_token

        # TODO: @jiawei.chen add prepare mel

        if pre_utt["lab"] is not None:
            prompt_lab = "\n".join([prompt_lab, pre_utt["lab"]])

        # prepare text-ids.
        ## 续写
        if self.hparams.icl_mode == "continuation":
            semantic_batch["lyrics"] = prompt_lab + "|" + infer_lab
        else:
            semantic_batch["lyrics"] = infer_lab  # 注意这里改了
        semantic_batch = self.lyrics_token_transform(semantic_batch)
        semantic_batch = self.add_conditions_transform(semantic_batch)
        # batching.
        for key in [
            "lyrics_tokens",
            "prompt_text_lens",
            "phones",
            "tones",
            "wordsegs",
            # cfg config
            "phones_cfg",
            "tones_cfg",
            "wordsegs_cfg",
        ]:
            semantic_batch[key] = semantic_batch[key].unsqueeze(0)
        # language.
        semantic_batch["src_lang"] = torch.LongTensor(
            [self.lang2id.get(self.hparams.src_lang)]
        )
        semantic_batch["tgt_lang"] = torch.LongTensor(
            [self.lang2id.get(self.hparams.tgt_lang)]
        )

        with (
            torch_allow_tf32(enable_matmul=False),
            torch.autocast(
                device_type="cuda", dtype=self.semantic_precision, enabled=True
            ),
        ):
            infer_umm_token = self.semantic_model.predict(
                semantic_batch, self.semantic_ar_model_hp
            )
        return infer_umm_token

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        set_seed(self.hparams.seed)
        uttid, prompt_lab, prompt_text, prompt_wav_path, infer_labs, infer_texts = batch

        if not isinstance(infer_labs, list):
            infer_labs = [infer_labs]
            infer_texts = [infer_texts]

        prompt_wav = self.preprocess_prompt_wav(
            prompt_wav_path,
            self.device,
            use_pad=False if self.hparams.version == "merge_v1" else True,
        )

        infer_meta_lst = os.path.join(self.hparams.output_path, "../meta_split.lst")
        pre_utt = {"lab": None, "token": None, "lang": None, "text": None}
        for i, (infer_lab, infer_text) in enumerate(zip(infer_labs, infer_texts)):

            if self.hparams.version == "merge_v1":
                infer_umm_token = self.text2semantic_v1(
                    prompt_wav, prompt_lab, prompt_text, infer_lab, infer_text, pre_utt
                )
            else:
                infer_umm_token = self.text2semantic_v2(
                    prompt_wav, prompt_lab, prompt_text, infer_lab, infer_text, pre_utt
                )
            lang = self.get_lab_lang(infer_lab)

            if self.hparams.use_pre_utt:  # use pre utt to continue predict
                pre_utt = {
                    "lab": infer_lab,
                    "token": infer_umm_token,
                    "lang": lang,
                    "text": infer_text,
                }

            uttid_split = f"{uttid}_split{i:04d}"
            if infer_umm_token is not None:
                infer_umm_token = infer_umm_token.detach().cpu().numpy()
                np.save(
                    os.path.join(self.hparams.output_path, f"{uttid_split}.npy"),
                    infer_umm_token,
                    allow_pickle=False,
                )
                with open(infer_meta_lst, "a+", encoding="utf-8") as f_w:
                    f_w.write(
                        "|".join(
                            [uttid_split, prompt_text, prompt_wav_path, infer_text]
                        )
                        + "\n"
                    )

    def get_lab_lang(self, tacolab):
        # zh, en, zh_en
        tacolab = tacolab.split("\n")
        if len(tacolab[0].split("\t")) != 5:
            if (
                tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
                or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
            ):
                tacolab = tacolab[1:]
        prefix_phn_list = [x.split("\t")[0][:2] for x in tacolab]
        if "C0" in prefix_phn_list:
            lang = "zh"
        else:
            lang = "en"
        return lang
