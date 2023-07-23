import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np

from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm
import math

from samantha.utils.hparams import DotDict
from recipes.bark.lit_modules.sample import sample
from recipes.text2semantic.lit_modules.lit_valle_coarse import sequence_mask
from s3a.providers.ctiga.utils.generation import InferenceParams
from samantha.utils.model_metric import ModelMetric

class VAET2SModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        logits_criterion_cls,
        dense_criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        tokenizer_len=50277,
        n_semantic=8192,
        checkpointing=True,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.logits_criterion = logits_criterion_cls()
        self.dense_criterion = dense_criterion_cls()
        self.tokenizer_len = tokenizer_len
        self.n_semantic = n_semantic
        self.extra_params = DotDict(extra_params)
        self.requires = {}

        # hugging face setting
        if hasattr(self.model, "resize_token_embeddings"):
            print("Resize token embeddings...")
            self.model.resize_token_embeddings(tokenizer_len + n_semantic + 2)
            self.model.config.use_cache = False
        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        self.model_metric = ModelMetric(
            precision=self.trainer.precision,
            model_obj_or_objs=self.model,
        )

        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, False)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):

        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                text_ids, text_id_lens, bns, bn_lens, input_tokens, seq_lens, pos_ids, seq_sen_ids, _, _ = batch
                b, t = input_tokens.shape
                loss_mask = sequence_mask(seq_lens, max_len=t, device="cuda")
                _, text_t = text_ids.shape
                text_loss_mask = sequence_mask(text_id_lens+2, max_len=text_t+2, device="cuda")
                text_loss_mask = F.pad(text_loss_mask, (0, t-text_loss_mask.shape[1]), "constant", 0)

                loss_mask = loss_mask - text_loss_mask
                org_len = input_tokens.size(1)

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            ret_dict, _ = self.model(text_ids, text_id_lens, bns, bn_lens, input_tokens)
            logits = ret_dict["logits"]
            dense = ret_dict["dense"]

        # if self.local_rank == 0:
        #     print("VALLE: Coarse Training.")

        pred_logits = logits[:, 0:org_len - 1, :]
        pred_dense = dense[:, 0:org_len - 1, :]

        bsz, bn_t, bn_c = bns.shape
        targets_dense = []
        for i in range(bsz):
            targets_dense.append(
                F.pad(bns[i, :bn_lens[i], :], (0, 0, text_id_lens[i]+2, t-(bn_lens[i]+text_id_lens[i]+2)), "constant", 0)
            )

        targets_dense = torch.stack(targets_dense)[:, 1:org_len, :]

        target_m, target_logs = torch.split(targets_dense, bn_c//2, dim=-1)
        pred_m, pred_logs = torch.split(pred_dense, bn_c//2, dim=-1)

        kl_loss = self.dense_criterion(pred_m, pred_logs, 
            target_m.detach(), target_logs.detach(), z_mask=loss_mask[:, 1:])

        # print('target: ', targets_dense[0][text_id_lens[0]: seq_lens[0]])
        # print('pred_dense: ', pred_dense[0][text_id_lens[0]: seq_lens[0]])
        # print('loss mask: ', loss_mask[0, 1:][text_id_lens[0]: seq_lens[0]])
        
        targets_logits = input_tokens[:, 1:org_len]
        ce_loss = self.logits_criterion(pred_logits.float(), targets_logits, mask=loss_mask[:, 1:])
        accu = ((pred_logits.argmax(dim=-1) == targets_logits).float() * loss_mask[:, 1:]).sum() / loss_mask[:, 1:].sum() * 100
        total_loss = kl_loss + ce_loss
        batch_tokens = b * t
        self.log_dict(
            {
                "kl_loss": kl_loss.item(),
                "ce_loss": ce_loss.item(),
                "loss": total_loss.item(),
                "accu": accu.item(),
                "bsz": b,
                "seqlen": t,
                "batch_tokens": batch_tokens,
            },
            prog_bar=True,
            sync_dist=True
        )
        self.model_metric.update(
            num_tokens=batch_tokens,
            stage=self.trainer.state.stage,
            model_kwargs=dict(batch_size=b, seqlen=t),
        )
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            metric = self.model_metric.compute(self.trainer.global_step)
            self.log_dict(
                metric, sync_dist=True, prog_bar=True
            )

        return total_loss

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
    def prepare_feature(self, wavs_16k, wavs_24k, texts, wav_16k_lens, wav_24k_lens, text_lens):
        device = wavs_16k.device

        # pad text to 256
        if texts.size(1) >= 256:
            texts = texts[:, 0:256]
        else:
            texts = torch.nn.functional.pad(texts, (0, 256 - texts.size(1)))

        semantic_codes, semantic_length = self.get_semantic_codes(wavs_16k, wav_16k_lens)
        # add semantic bos token and eos token
        semantic_codes = torch.nn.functional.pad(semantic_codes, (1, 1))
        # bos
        semantic_codes[:, 0] = self.n_semantic
        # eos
        semantic_mask = torch.arange(semantic_codes.size(1)).unsqueeze(0).to(device) < (1 + semantic_length.unsqueeze(1))
        semantic_codes = torch.where(
            semantic_mask,
            semantic_codes,
            torch.zeros_like(semantic_codes) + self.n_semantic + 1,
        )
        # offset: BPE
        semantic_codes += self.tokenizer_len

        # concat input_tokens
        input_tokens = torch.cat([texts, semantic_codes], dim=1)

        # loss mask
        text_mask = torch.arange(texts.size(1)).unsqueeze(0).to(device) < (1 + text_lens.unsqueeze(1)) # add eos loss
        semantic_mask = torch.arange(semantic_codes.size(1)).unsqueeze(0).to(device) < (2 + semantic_length.unsqueeze(1))
        loss_mask = torch.cat([text_mask, semantic_mask], dim=1).float()
        return input_tokens, loss_mask

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
    def inference_from_text(self, batch, tokenizer):
        
        text_ids, text_id_lens, bns, bn_lens, seqs, seq_lens, pos_ids, seq_sen_ids, utts = batch
        b, t = seqs.shape   
        attention_mask = sequence_mask(seq_lens, max_len=None, device=seqs.device)
        input_tokens = seqs
        # seqs = (
        #         [self.tokenizer.bos]
        #         + list(text_id)
        #         + [self.tokenizer.sep]
        #         + [0] * bn_T # place holder
        #     )
        # inference
        semantic_outputs = []

        
        # llama style inference
        self.model.params.use_cache = True
        start_pos = 0
        
        inference_params = InferenceParams(
            max_sequence_len=8000, 
            max_batch_size=b,
            fused_ft_kernel=False
        )
        z_list = []
        with torch.autocast(device_type="cuda", enabled=True):
            for i in tqdm(range(3000)):
                if i == 0:
                    model_outputs, bn_in_z = self.model(text_ids, text_id_lens, 
                        bns, bn_lens, input_tokens, start_pos=start_pos, inference_params=inference_params)
                else:
                    model_outputs, bn_in_z = self.model(None, None, 
                        bns, None, input_tokens, start_pos=start_pos, use_cache=True, inference_params=inference_params)
                    z_list.append(bn_in_z)
                logits = model_outputs["logits"]
                pred_logits = logits[:, -1, :]
                # pred_logits[:, 1:self.tokenizer_len + self.n_semantic + 2] = -1e3
                # if i < 50:
                #     pred_logits[:, self.tokenizer_len + self.n_semantic + 2] = -1e5
                # samples = sample(pred_logits, temp=0.9, mode="naive", device=seqs.device) # [b, 1]
                samples = torch.argmax(pred_logits)
                # print(i , ' ', samples, '  ', self.tokenizer_len, '  ', self.n_semantic)
                if i > 10 and samples.item() == self.tokenizer_len + self.n_semantic + 2:
                    break
                pred_dense = model_outputs["dense"][:, -1:, :]
                # print(i, ' ', input_tokens.shape, ' ', text_id_lens, ' ', bn_lens)

                start_pos += input_tokens.size(1)
                inference_params.sequence_len_offset = start_pos

                # next infer
                semantic_outputs.append(pred_dense)
                input_tokens = torch.zeros([1, 1], dtype=torch.int).to(pred_dense.device)
                bns = pred_dense
                pos_ids = torch.cat([pos_ids, pos_ids[:, -1:] + 1], dim=1)  # [b, t]
                seq_sen_ids = torch.cat(
                    [seq_sen_ids, torch.zeros_like(seq_sen_ids[:, -1:]) + 2], dim=1
                )

        z_outputs = torch.cat(z_list[1:], dim=1) # [b, t, c]
        semantic_outputs = torch.cat(semantic_outputs, dim=1) # [b, t, c]

        # semantic_outputs = torch.cat([seqs, semantic_outputs], dim=1)
        self.model.params.use_cache = False

        return z_outputs, semantic_outputs, pos_ids, seq_sen_ids






