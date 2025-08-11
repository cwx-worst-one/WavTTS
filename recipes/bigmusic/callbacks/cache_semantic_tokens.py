import pytorch_lightning as pl
import torch
import os
from typing import Any, List, Union
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

class CacheSemanticTokensCallback(pl.Callback):
    def __init__(
        self,
        beam_size=1,
        samples_to_save=1,
        token_padding=32768,
        cache_dir=None
    ):
        super().__init__()
        self.beam_size = beam_size
        self.samples_to_save = samples_to_save
        self.cache_dir = cache_dir # TODO: use cache dir
        self.token_padding = token_padding

    def on_predict_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # load tokens from directory
        if self.cache_dir is None:
            output_dir = pl_module.extra_params.output_dir
        else:
            output_dir = self.cache_dir
        all_semantic_tokens = []
        for i, _ in enumerate(batch["index"]):
            semantic_tokens_fp = get_file_name(output_dir, batch, i, sample_round=dataloader_idx)
            if Path(semantic_tokens_fp).exists():
                print("Loading from cached semantic tokens.", semantic_tokens_fp)
                semantic_tokens = torch.load(semantic_tokens_fp)
                all_semantic_tokens.append(semantic_tokens)

        if len(all_semantic_tokens) == len(batch["index"]):
            all_semantic_tokens = pad_sequence(all_semantic_tokens, batch_first=True, padding_value=self.token_padding)
            batch["semantic_tokens"] = all_semantic_tokens.to(pl_module.device)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.cache_dir is None:
            output_dir = pl_module.extra_params.output_dir
        else:
            output_dir = self.cache_dir
        for i, semantic_tokens in enumerate(outputs["generated_semantic_tokens"]):
            semantic_tokens_fp = get_file_name(output_dir, batch, i, sample_round=dataloader_idx)
            if Path(semantic_tokens_fp).exists(): continue
            Path(semantic_tokens_fp).parent.mkdir(parents=True, exist_ok=True)
            semantic_tokens = torch.save(semantic_tokens, semantic_tokens_fp)
        if pl_module.extra_params.get('cached_sementic_tokens_to_hdfs', False):
            print("Caching semantic tokens to hdfs.", output_dir, pl_module.extra_params.hdfs_output_dir)
            os.system(f"hdfs dfs -put -f {output_dir} {pl_module.extra_params.hdfs_output_dir}")

def get_file_name(output_dir, batch, idx, beam_size=1, sample_round=0, samples_to_save=1):
    index = batch.get('index')
    categories = batch.get('category', '')
    if not categories:
        categories = batch.get('text_category', '')

    
    prompt_idx = idx // beam_size
    beam_idx = idx % beam_size
    ii = prompt_idx if sample_round == 0 else prompt_idx // sample_round
    if categories is not None and categories[prompt_idx]:
        wav_dir = os.path.join(output_dir, categories[ii])
    else:
        wav_dir = output_dir
    # os.makedirs(wav_dir, exist_ok=True)
    file_name = ""
    file_name = index[ii]
    wav_file_name = file_name
    if sample_round > 0:
        wav_file_name += ("_r" + str(prompt_idx % sample_round))
    if samples_to_save > 1 and beam_size > 1:
        wav_file_name += f".{beam_idx}"

    semantic_tokens_fp = os.path.join(wav_dir, f"{wav_file_name}.semantic_tokens.pt")
    return semantic_tokens_fp
