import json
from pathlib import Path
from typing import Any, Optional, Union

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio

from recipes.bigmusic.callbacks.common_callbacks import UploadToEasyCycleCallback

from .utils import sync_all_ranks


class SaveOutputsCallback(pl.Callback):
    def __init__(
        self,
        sample_rate: int,
        output_dir: str,
        index_key: str = "index",
        category_key: str = "category",
        output_audio_key: str = "generated_audio",
        output_token_key: str = "generated_semantic_tokens",
        additional_keys: Optional[list[str]] = None,
        default_category: str = "default",
        save_semantic_tokens: bool = False,
        save_audio: bool = True,
        upload_audio: bool = True,
        save_meta: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        # saving modes
        self.save_semantic_tokens = save_semantic_tokens
        self.save_audio = save_audio
        self.upload_audio = upload_audio
        self.save_meta = save_meta
        # keys
        self.output_audio_key = output_audio_key
        self.output_token_key = output_token_key
        self.index_key = index_key
        self.category_key = category_key
        self.additional_keys = [] if not additional_keys else additional_keys

        self.default_category = default_category

        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Write audios in batch to disk."""
        # This is redundant since this file already exists after finishing the first batch.
        # But it helps us save the file as soon as we complete the first batch's inference.
        with open(
            Path(self.output_dir) / f"inference_params.{trainer.global_rank}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(pl_module.extra_params, f, indent=2)

        save_batch_outputs(
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            self_output_dir=self.output_dir,
            sample_rate=self.sample_rate,
            sample_round=dataloader_idx,
            default_category=self.default_category,
            # keys
            index_key=self.index_key,
            category_key=self.category_key,
            output_audio_key=self.output_audio_key,
            output_token_key=self.output_token_key,
            additional_keys=self.additional_keys,
            # saving modes
            save_semantic_tokens=self.save_semantic_tokens,
            save_audio=self.save_audio,
            upload_audio=self.upload_audio,
            save_meta=self.save_meta,
        )

    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not sync_all_ranks(trainer, self.output_dir, self.__class__.__name__):
            return
        metadata_fps = list(Path(self.output_dir).glob(f"**/*.metadata.json"))
        if len(metadata_fps) == 0:
            return
        index_fname = Path(self.output_dir) / "index.csv"
        with open(index_fname, "w", encoding="utf-8") as fw:
            fw.write("file_name,index,audio_url\n")
            for fp in metadata_fps:
                with open(fp, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                file_name = metadata["file_name"]
                index = metadata["index"]
                audio_url = metadata.get("audio_url", "")
                fw.write(f"{file_name},{index},{audio_url}\n")
        print(f"Wrote index to {index_fname}")


def save_batch_outputs(
    outputs: dict[str, Any],  # inference output
    batch: dict[str, Any],  # inference batch batch
    batch_idx: int,
    self_output_dir: Union[str, Path],
    sample_rate: int,
    sample_round: int = 0,
    # keys
    index_key: str = "index",
    category_key: str = "category",
    output_audio_key: str = "generated_audio",
    output_token_key: str = "generated_semantic_tokens",
    additional_keys: Optional[list[str]] = None,
    default_category: str = "default",
    # saving modes
    save_semantic_tokens: bool = False,
    save_audio: bool = True,
    upload_audio: bool = True,
    save_meta: bool = True,
):
    """
    outputs: generated_audio, generated_audio_tensor, generated_semantic_tokens
    batch: ???
    """
    wavs = outputs[output_audio_key]
    batch_size = len(wavs)  # wavs is a list

    semantic_tokens = outputs.get(output_token_key, [None] * batch_size)

    indices = batch.get(index_key, [f"{batch_idx}_{ii}" for ii in range(batch_size)])
    categories = batch.get(category_key, [default_category] * batch_size)
    extra_items = (
        {k: batch.get(k, [None] * batch_size) for k in additional_keys}
        if additional_keys
        else {}
    )
    output_paths = []

    # sample_idx: the number of sample in the batch
    # data_idx: the index string in the dataset
    for sample_idx, (data_idx, category, wav) in enumerate(
        zip(indices, categories, wavs)
    ):

        # To simplify the logic, always generate a meta dict (metadata) first,
        # even if the user chooses to save nothing.
        sample_idx = sample_idx if sample_round == 0 else sample_idx // sample_round
        wav_dir = Path(self_output_dir) / category
        wav_dir.mkdir(parents=True, exist_ok=True)

        file_name = data_idx
        if sample_round > 0:
            file_name += "_r" + str(sample_idx % sample_round)

        if not extra_items:
            sample_extra_items = {}
        else:
            sample_extra_items = {k: extra_items[k][sample_idx] for k in extra_items}

        metadata = {
            "index": data_idx,
            "round": sample_round,
            "file_name": file_name,
            **sample_extra_items,
        }

        if save_audio or upload_audio:
            wav_fp = Path(wav_dir) / f"{file_name}.generated.wav"
            print(f"[Saving] {wav_fp}")
            save_wav(wav.cpu().float(), wav_fp, sr=sample_rate)
            output_paths.append(wav_fp)

            if upload_audio:
                metadata["audio_url"] = UploadToEasyCycleCallback.upload_file(wav_fp)
                print(f"[Saving] {Path(wav_fp).name}: {metadata['audio_url']}")

            if not save_audio:
                Path(wav_fp).unlink()

        if save_semantic_tokens and semantic_tokens is not None:
            semantic_tokens_fp = Path(wav_dir) / f"{file_name}.semantic_tokens.pt"
            torch.save(semantic_tokens[sample_idx], semantic_tokens_fp)

        if save_meta:
            print("Saving metadata", metadata)
            meta_fp = Path(wav_dir) / f"{file_name}.metadata.json"
            with open(meta_fp, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

    return output_paths


def save_wav(audio: torch.Tensor, output_file: Union[str, Path], sr: int = 24000):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(output_file), audio, sr)


def load_wav(path, sr: int = 24000, mono: bool = True):
    wav, sr = librosa.load(str(path), sr=sr, mono=mono)
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2_147_483_648.0
    return wav
