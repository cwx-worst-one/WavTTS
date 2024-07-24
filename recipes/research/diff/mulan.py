from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from julius.resample import resample_frac

from recipes.musiclm.requires.mulan.mulan_infer_sstk_mae import (
    create_mulan_model,
    mulan_inference,
    mulan_rvq_indexs,
)
from samantha.models.base import LightningModuleBase


@dataclass
class MulanResult:
    mulan_text_embs: torch.Tensor
    mulan_audio_embs: torch.Tensor
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor


class Mulan(LightningModuleBase):

    def __init__(
        self,
        output_type: str,
        text_only: bool = False,
        text_feature_layer_idx: int = -1,
    ):
        super().__init__()

        assert output_type in ["cls", "seq"]
        self.output_type = output_type
        self.text_only = text_only
        self.text_feature_layer_idx = text_feature_layer_idx

        # everything in shutterstock:
        # ckpt_path = "/mnt/bn/audio-diffusion/peng/clap_sstk_mulan_files/MuLan_sstk/CLAP_all_text_space_connect_large_dropout/checkpoints/mulan-step=005000-median_rank_1=61-kaggle.ckpt"

        # genre-balanced:
        ckpt_path = "/mnt/bn/audio-diffusion/peng/clap_sstk_mulan_files/MuLan_sstk/CLAP_genre_balanced/checkpoints/mulan-step=005000-median_rank_1=72-kaggle.ckpt"

        # sequence-level mulan:
        self.model = create_mulan_model(
            ckpt_path,
            device="cpu",
            version="v2",
            strict=False,
            output_type=output_type,
            text_feature_layer_idx=text_feature_layer_idx,
        )
        self.rvq_fn = mulan_rvq_indexs
        self.inference_fn = mulan_inference
        self.sample_rate = 24000
        self.max_duration = 10
        self.frame_rate = 50  # audio tower: seq
        self.n_channels = 1

        if text_only and text_feature_layer_idx != -1:
            self.n_embd = 768
        else:
            self.n_embd = 512

        if text_only:
            del self.model.music_encoder

    def cast_to_rank(self, rank: int):
        self.model.to(f"cuda:{rank}")

        if not self.text_only:
            self.model.music_encoder.mut.manually_to_device(f"cuda:{rank}")

    def forward(
        self,
        text: Optional[List[str]] = None,
        audio: Optional[torch.Tensor] = None,
        sample_rate: Optional[int] = None,
        shift_seconds: int = 10,
    ):
        assert text is None or audio is None

        if audio is not None:
            audio = audio.mean(dim=1, keepdim=True)
            if sample_rate != self.sample_rate:
                audio = resample_frac(audio, sample_rate, self.sample_rate)

            # mulan_audio_embs = []
            # audios = audio.split(self.sample_rate * self.max_duration, dim=2)
            # for a in audios:
            #     mulan_embs = self.inference(a)
            #     mulan_audio_embs.append(mulan_embs)
            # mulan_audio_embs2 = torch.cat(mulan_audio_embs, dim=1)

            b, c, t = audio.shape
            max_frames = self.sample_rate * self.max_duration

            if audio.shape[2] < max_frames:
                audio = F.pad(audio, (0, max_frames - audio.shape[2]), value=0.0)

            audio = audio.unfold(
                2,
                self.sample_rate * self.max_duration,
                self.sample_rate * shift_seconds,
            )
            audio = audio.reshape(-1, c, self.sample_rate * self.max_duration)
            mulan_audio_embs = self.model.music_encoder(audio)
            mulan_audio_embs = mulan_audio_embs.reshape(b, -1, self.n_embd)  # [B, T, D]
            # torch.testing.assert_close(mulan_audio_embs, mulan_audio_embs2)
            return MulanResult(
                mulan_text_embs=None,
                hidden_states=None,
                attention_mask=None,
                mulan_audio_embs=mulan_audio_embs,
            )
        else:
            text_result = self.model.encode_text(text, return_dict=True)
            mulan_text_embs = text_result["text_embed"].unsqueeze(dim=1)  # [B, 1, D]
            # hidden_states = text_result["last_hidden_state"]
            hidden_states = text_result["hidden_states"]
            attention_mask = text_result["attention_mask"]
            return MulanResult(
                mulan_text_embs=mulan_text_embs,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                mulan_audio_embs=None,
            )


if __name__ == "__main__":
    sample_rate = 44100
    mulan = Mulan(output_type="seq", text_only=True, text_feature_layer_idx=-2)
    mulan = mulan.to("cuda")
    mulan.eval()
    mulan.freeze()

    with torch.no_grad():
        result = mulan.forward(
            text=[
                "I'm free as a bird. I'm free as a bird. I'm free as a bird. I'm free as a bird. I'm free as a bird. I'm free as a bird. I'm free as a bird. "
            ],
            audio=None,
            sample_rate=sample_rate,
        )
    # print(result.avg_hidden_states.mean(), result.avg_hidden_states.var())

    assert result.hidden_states.shape == (1, 400, 768)
    assert result.mulan_text_embs.shape == (1, 1, 512)
