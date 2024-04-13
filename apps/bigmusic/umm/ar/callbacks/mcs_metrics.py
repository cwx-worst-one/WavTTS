import json
from pathlib import Path

import pytorch_lightning as pl
import torch

from apps.bigmusic.umm.ar.datasets.transforms.lyrics_segment import (
    crop_pad_to_seq_length,
)
from apps.bigmusic.umm.ar.utils.utils import load_wav
from apps.bigmusic.umm.ar.lightning.embedding_modules import get_mulan_embeds
from apps.bigmusic.umm.ar.utils.format_utils import update_json


class MCSMetricsCallback(pl.Callback):
    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        requires = pl_module.semantic_module.requires
        sample_rate = pl_module.extra_params.sample_rate
        if "output_paths" in pl_module.extra_params:
            generated_output_fps = pl_module.extra_params.output_paths
        else:
            output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob("**/*.generated.wav"))
        run_mcs_metrics(
            requires,
            generated_output_fps,
            device=pl_module.device,
            sample_rate=sample_rate,
        )


def run_mcs_metrics(requires, generated_output_fps, device="cuda", sample_rate=24000):
    mulan_min_duration = 10 * sample_rate

    def _load_audio_tensor(audio_path):
        wav_tensor = torch.tensor(load_wav(str(audio_path), sr=sample_rate)).to(device)
        if wav_tensor.shape[-1] < mulan_min_duration:
            wav_tensor = crop_pad_to_seq_length(wav_tensor, mulan_min_duration)
        return wav_tensor.unsqueeze(0)

    for idx, generated_output_fp in enumerate(generated_output_fps):
        metadata_fp = str(generated_output_fp).replace("generated.wav", "metadata.json")
        with open(metadata_fp, "r") as f:
            metadata = json.load(f)
        conditions = metadata["conditions"]
        # Compute MCS based on wavs
        if "style_text" in conditions:
            gt_emb = get_mulan_embeds(
                requires, metadata["style_text"], data_type="text"
            ).to(device)
        elif "style_audio" in conditions:
            style_audio_fp = str(generated_output_fp).replace(
                ".generated.wav", ".style_audio.wav"
            )
            wav_style = _load_audio_tensor(style_audio_fp)
            gt_emb = get_mulan_embeds(requires, wav_style, data_type="music").to(device)
        else:
            print(
                "Could not calculate MCS metrics. Could not find style_text or style_audio in conditions",
                conditions,
            )
            return
        wav_gen = _load_audio_tensor(generated_output_fp)
        audio_emb = get_mulan_embeds(requires, wav_gen, data_type="music").to(device)
        mcs = torch.nn.functional.cosine_similarity(gt_emb, audio_emb).cpu().item()
        update_json(metadata_fp, {"mcs": round(mcs, 3)})
