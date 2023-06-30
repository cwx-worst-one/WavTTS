import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F

from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm

from samantha.utils.hparams import DotDict
from ...utils.sample import sample
from recipes.valle.lit_modules.lit_valle_coarse import sequence_mask
from samantha.models.ctiga_llama import create_ctiga_llama
from samantha.utils.model_metric import ModelMetric
from recipes.valle.datasets import PhoneTokenizerWithAudioTokens
from s3a.providers.ctiga.utils.generation import InferenceParams


class ValleCoarseWdsModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        tokenizer_len=50277,
        n_semantic=8192,
        checkpointing=True,
        provider="default",
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert provider in ["default", "ctiga"]
        self.provider = provider
        if self.provider == "ctiga":
            print("Use ctiga as self.model.")
            self.model = create_ctiga_llama(model_cls)
        else:
            self.model = model_cls()
        self.criterion = criterion_cls()
        self.tokenizer_len = tokenizer_len
        self.n_semantic = n_semantic
        self.extra_params = DotDict(extra_params)
        self.requires = {}

        torch._C._jit_set_bailout_depth(0)

        # hugging face setting
        if hasattr(self.model, "resize_token_embeddings"):
            print("Resize token embeddings...")
            self.model.resize_token_embeddings(tokenizer_len + n_semantic + 2)
            self.model.config.use_cache = False
        if checkpointing:
            self.model.gradient_checkpointing_enable()

        self.tokenizer = PhoneTokenizerWithAudioTokens(
            self.extra_params.phone_tokens_num, self.extra_params.audio_tokens_num
        )

    def setup(self, stage: str) -> None:
        # mfu metric
        self.metric = ModelMetric(
            precision=self.trainer.precision,
            model_obj_or_objs=self.model,
        )
        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            utt_ids, seqs, seq_lens, pos_ids, seq_sen_ids, full_seqs = batch
            b, t = seqs.shape

            # update mfu
            self.metric.update(
                num_tokens=b * t,
                stage=self.trainer.state.stage,
                model_kwargs={"batch_size": b, "seq_len": t}
            )
            if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
                self.log_dict(
                    self.metric.compute(self.trainer.global_step),
                    prog_bar=True,
                    sync_dist=True,
                )

            loss_mask = sequence_mask(seq_lens, max_len=t, device="cuda")
            logits = self.model(seqs).logits

        x = logits[:, 0:t - 1, :]
        targets = seqs[:, 1:t]
        loss = self.criterion(x.float(), targets, mask=loss_mask[:, 1:])
        accu = ((x.argmax(dim=-1) == targets).float() * loss_mask[:, 1:]).sum() / loss_mask[:, 1:].sum() * 100
        self.log_dict(
            {
                "ar_train_loss": loss.item(),
                "accu": accu.item(),
                "bsz": b,
                "avg_tokens": sum(seq_lens)/b,
                "max_tokens": max(seq_lens)
            },
            prog_bar=True,
            sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

    @torch.no_grad()
    def get_semantic_codes(self, wavs_16k, wavs_16k_len):
        infer_fn = self.requires["semantic_infer_fn"]
        semantic_codes, semantic_length = infer_fn(
            self.requires["semantic"],
            self.requires["centroids"],
            wavs_16k,
            wavs_16k_len,
            wavs_16k.device,
        )
        return semantic_codes, semantic_length

    @torch.no_grad()
    def get_codec_codes(self, x):
        output = self.requires["ss"](x)[2]
        output = torch.stack(output, dim=2) # [b, t, n_code]
        return output

    @torch.no_grad()
    def inference_from_text(self, batch, tokenizer, temperature=0.9, thres=0.9):
        seqs, seq_lens, pos_ids, seq_sen_ids, utts = batch
        b, t = seqs.shape   
        attention_mask = sequence_mask(seq_lens, max_len=None, device=seqs.device)
        input_tokens = seqs

        # inference
        semantic_outputs = []

        if hasattr(self.model, "resize_token_embeddings"):
            # hugging face style
            past_key_values = None
            for i in tqdm(range(1000)):
                model_outputs = self.model(input_tokens, past_key_values=past_key_values, use_cache=True)
                logits = model_outputs["logits"]
                past_key_values = model_outputs["past_key_values"]
                pred_logits = logits[:, -1, self.tokenizer_len: self.tokenizer_len + self.n_semantic + 2]
                pred_logits[:, self.n_semantic] = -1e3
                samples = sample(pred_logits, temp=0.9, mode="naive", device=seqs.device) # [b, 1]
                if samples.item() == self.n_semantic + 1:
                    break
                # next infer
                semantic_outputs.append(samples)
                input_tokens = samples + self.tokenizer_len # offset: token
            semantic_outputs = torch.cat(semantic_outputs, dim=1) # [b, t]
        else:
            # llama style inference
            max_length = 8000 - input_tokens.size(1)
            if self.provider == "default":
                self.model.params.use_cache = True
                start_pos = 0
                for i in tqdm(range(max_length)):
                    model_outputs = self.model(input_tokens, start_pos=start_pos)
                    logits = model_outputs["logits"]
                    start_pos += input_tokens.size(1)
                    pred_logits = logits[:, -1, self.tokenizer_len + 1: self.tokenizer_len + self.n_semantic + 3]
                    pred_logits[:, self.n_semantic] = -1e3
                    if i < 50:
                        pred_logits[:, self.n_semantic + 1] = -1e5
                    samples = sample(pred_logits, temp=0.9, mode="naive", device=seqs.device) # [b, 1]
                    if samples.item() == self.n_semantic + 1:
                        break
                    semantic_outputs.append(samples + self.tokenizer_len + 1)
                    input_tokens = samples + self.tokenizer_len + 1 # offset: token
                    pos_ids = torch.cat([pos_ids, pos_ids[:, -1:] + 1], dim=1)  # [b, t]
                    seq_sen_ids = torch.cat(
                        [seq_sen_ids, torch.zeros_like(seq_sen_ids[:, -1:]) + 2], dim=1
                    )

                semantic_outputs = torch.cat(semantic_outputs, dim=1) # [b, t]
                semantic_outputs = torch.cat([seqs, semantic_outputs], dim=1)
                self.model.params.use_cache = False
            elif self.provider == "ctiga":
                batch_size = 1
                inference_params = InferenceParams(
                        max_sequence_len=8000, 
                        max_batch_size=batch_size,
                        fused_ft_kernel=False)
                pred_logits = self.model(
                        input_tokens, 
                        inference_params=inference_params, 
                        last_token_only=True
                    ).logits
                pred_logits = pred_logits[..., 
                                self.tokenizer_len + 1: self.tokenizer_len + self.n_semantic + 3]
                pred_logits[:, self.n_semantic] = -1e3
                pred_logits[:, self.n_semantic + 1] = -1e3
                samples = sample(pred_logits, 
                        temp=temperature, 
                        thres=thres,
                        mode="naive", 
                        device=seqs.device) # [b, 1]
                next_token = samples + self.tokenizer_len + 1
                semantic_outputs.append(next_token)
                inference_params.sequence_len_offset = input_tokens.shape[1]
                pos_ids = torch.cat([pos_ids, pos_ids[:, -1:] + 1], dim=1)  # [b, t]
                seq_sen_ids = torch.cat(
                    [seq_sen_ids, torch.zeros_like(seq_sen_ids[:, -1:]) + 2], dim=1
                )

                for i in tqdm(range(max_length - 1)):
                    position_ids = torch.full(
                        (batch_size, 1), 
                        inference_params.sequence_len_offset,
                        dtype=torch.long, 
                        device=input_tokens.device)
                    pred_logits = self.model(
                        next_token,
                        position_ids=position_ids,
                        inference_params=inference_params, 
                        last_token_only=True).logits
                    pred_logits = pred_logits[..., self.tokenizer_len + 1: self.tokenizer_len + self.n_semantic + 3]
                    pred_logits[:, self.n_semantic] = -1e3
                    if i < 50:
                        pred_logits[:, self.n_semantic + 1] = -1e3
                    samples = sample(pred_logits, 
                        temp=temperature, 
                        thres=thres,
                        mode="naive", 
                        device=seqs.device) # [b, 1]
                    if samples.item() == self.n_semantic + 1:
                        break
                    next_token = samples + self.tokenizer_len + 1
                    semantic_outputs.append(next_token)
                    inference_params.sequence_len_offset += 1
                    
                    pos_ids = torch.cat([pos_ids, pos_ids[:, -1:] + 1], dim=1)  # [b, t]
                    seq_sen_ids = torch.cat(
                        [seq_sen_ids, torch.zeros_like(seq_sen_ids[:, -1:]) + 2], dim=1
                    )

                semantic_outputs = torch.cat(semantic_outputs, dim=1) # [b, t]
                semantic_outputs = torch.cat([seqs, semantic_outputs], dim=1)


        return semantic_outputs, pos_ids, seq_sen_ids
