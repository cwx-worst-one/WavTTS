import os
import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
from pytorch_lightning.profilers import PassThroughProfiler
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from pytorch_lightning.strategies import DeepSpeedStrategy

from samantha.utils.hparams import DotDict
from recipes.valle.datasets import PhoneTokenizerWithAudioTokens
from tqdm import tqdm

def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask

def compute_loss(logits, target, mask):
    logits = logits.contiguous()
    target = target.contiguous()
    mask = mask.contiguous()

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size()) * mask
    # mask: (batch, max_len)
    loss = losses.sum() / mask.sum()
    return loss

def compute_token_acc(logits, target, mask):
    logits = logits.contiguous()
    target = target.contiguous()
    mask = mask.contiguous()

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    token_acc_flat = torch.argmax(log_probs_flat, dim=-1, keepdim=True) == target_flat
    # losses: (batch, max_len)
    token_acc = token_acc_flat.view(*target.size()) * mask
    # mask: (batch, max_len)
    token_acc = token_acc.sum() / mask.sum()
    return token_acc

class ValleCoarse(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        resume_ckpt_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.tokenizer = PhoneTokenizerWithAudioTokens(
            self.extra_params.phone_tokens_num, self.extra_params.audio_tokens_num
        )
        self.requires = {}
        if checkpointing:
            self.model.gradient_checkpointing_enable()
        if resume_ckpt_path:
            state_dict = torch.load(resume_ckpt_path)['state_dict']
            new_state_dict = []
            new_state_dict = {k.replace("model.",""):v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
            print("Loading state_dict from {} successfully".format(resume_ckpt_path))
    def setup(self, stage: str) -> None:
        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        if self.hparams.required_modules:
            for name, (hpath, initializer) in self.hparams.required_modules.items():
                print(name)
                self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[LightningModule]LLMModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                wavs, text_ids, wav_lens, text_lens, utt_ids = batch
                seqs, seq_lens, pos_ids, seq_sen_ids = self.prepare_feature(
                    wavs, text_ids, wav_lens, text_lens, "cuda")
                b, t = seqs.shape
                attention_mask = sequence_mask(seq_lens, max_len=t, device="cuda")

        with self.profiler.profile("[LightningModule]LLMModule.model_forward"):
            # TODO: add location attention mask
            # global_step = self.trainer.global_step
            if "use_pos_ids" in self.extra_params and not self.extra_params.use_pos_ids:
                logits = self.model(input_ids=seqs,
                                    attention_mask=attention_mask)["logits"]

            else:
                logits = self.model(input_ids=seqs,
                                    attention_mask=attention_mask,
                                    position_ids=pos_ids)["logits"]
        # calculate loss
        loss_mask = sequence_mask(seq_lens - 1, max_len=seq_lens.max() - 1, device="cuda")
        # only calculate loss on output
        if self.extra_params.mask_input_for_train:
            for idx in range(b):
                input_len = torch.sum(seq_sen_ids[idx,:]==seq_sen_ids[idx,0])-1
                loss_mask[idx,:input_len] = 0
        loss = compute_loss(logits[:, 0:seq_lens.max()-1, :], seqs[:, 1:seq_lens.max()], mask=loss_mask)
        self.log_dict({
                "train_loss": loss,
                "batch_size": b
            },
            logger=True, sync_dist=True, prog_bar=True, batch_size=b)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     with torch.autocast(device_type="cuda", enabled=False):
    #         wavs, text_ids, wav_lens, text_lens, utt_ids = batch
    #         seqs, seq_lens, pos_ids, seq_sen_ids = self.prepare_feature(
    #             wavs, text_ids, wav_lens, text_lens, "cuda")
    #         b, t = seqs.shape
    #         attention_mask = sequence_mask(seq_lens, max_len=t, device="cuda")

    #     if "use_pos_ids" in self.extra_params and not self.extra_params.use_pos_ids:
    #         logits = self.model(
    #             input_ids=seqs, attention_mask=attention_mask
    #         )["logits"]
    #     else:
    #         logits = self.model(
    #             input_ids=seqs,
    #             attention_mask=attention_mask,
    #             position_ids=pos_ids
    #         )["logits"]

    #     # calculate loss
    #     loss_mask = sequence_mask(
    #         seq_lens - 1, max_len=seq_lens.max() - 1, device="cuda"
    #     )
    #     # only calculate loss on output
    #     if self.extra_params.mask_input_for_val:
    #         for idx in range(b):
    #             input_len = torch.sum(seq_sen_ids[idx, :] == seq_sen_ids[idx, 0]) - 1
    #             loss_mask[idx, :input_len] = 0

    #     token_acc = compute_token_acc(
    #         logits[:, 0 : seq_lens.max() - 1, :],
    #         seqs[:, 1 : seq_lens.max()],
    #         mask=loss_mask,
    #     )
    #     loss = compute_loss(
    #         logits[:, 0 : seq_lens.max() - 1, :],
    #         seqs[:, 1 : seq_lens.max()],
    #         mask=loss_mask,
    #     )
    #     self.log_dict(
    #         {
    #             "val_loss": loss,
    #             "val_token_acc": token_acc
    #         },
    #         logger=True,
    #         sync_dist=True,
    #         prog_bar=True,
    #         batch_size=b
    #     )
    #     return token_acc, loss

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False

    def configure_optimizers(self):
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(
                self.model.parameters(), lr=self.extra_params.lr, weight_decay=self.extra_params.weight_decay
            )
        elif isinstance(self.trainer.strategy, DeepSpeedStrategy):
            optimizer = FusedAdam(
                self.model.parameters(), lr=self.extra_params.lr, weight_decay=self.extra_params.weight_decay
            )
        else:
            optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def generate(self, batch, tokenizer):
        with torch.autocast(device_type="cuda", enabled=False):
            seqs, seq_lens, pos_ids, seq_sen_ids, utts = batch
            b, t = seqs.shape
            attention_mask = sequence_mask(seq_lens, max_len=None, device=seqs.device)
            temperature = 1.0
            output_tensor = torch.empty(seqs.shape[0], 0).to(seqs.device)
            for j in range(1500):
                if self.extra_params.use_pos_ids:
                    gpt2_lm_outputs = self.model(
                        input_ids=seqs, attention_mask=None, position_ids=pos_ids
                    )
                else:
                    gpt2_lm_outputs = self.model(
                        input_ids=seqs, attention_mask=None
                    )

                logits = gpt2_lm_outputs["logits"]
                logits = logits * temperature
                logits[:, :, : 1 + tokenizer.phone_token_num] = -1e5
                logits[:, :, tokenizer.sep] = -1e5
                if j < 50:
                    logits[:, :, tokenizer.eos] = -1e5
                probs = logits[:, -1, :].softmax(dim=-1)  # [b, d]
                prob_dist = torch.distributions.categorical.Categorical(probs=probs)
                samples = prob_dist.sample().unsqueeze(1).to(logits.device)  # [b, 1]
                # for next infer
                seqs = torch.cat([seqs, samples], dim=1)  # [b, t]
                pos_ids = torch.cat([pos_ids, pos_ids[:, -1:] + 1], dim=1)  # [b, t]
                seq_sen_ids = torch.cat(
                    [seq_sen_ids, torch.zeros_like(seq_sen_ids[:, -1:]) + 2], dim=1
                )

                if samples.item() == tokenizer.eos:
                    break
                elif j == 1499 or t + j == 2047:
                    # too long break
                    samples[:, :] = tokenizer.eos
                    seqs = torch.cat([seqs, samples], dim=1)  # [b, t]
                    pos_ids = torch.cat([pos_ids, pos_ids[:, -1:] + 1], dim=1)  # [b, t]
                    seq_sen_ids = torch.cat(
                        [seq_sen_ids, torch.zeros_like(seq_sen_ids[:, -1:]) + 2], dim=1
                    )
                    break
                else:
                    output_tensor = torch.cat([output_tensor, samples], dim=1)
        return seqs, pos_ids, seq_sen_ids, output_tensor

    @torch.no_grad()
    def prepare_feature(self, wavs, text_ids, wav_lens, text_lens, device):
        batch = len(wavs)
        text_ids = self.tokenizer.tokenize(text_ids, "inputs")
        wav_ids = self.tokenizer.tokenize(self.get_codec_codes(wavs), "targets")

        # wav_id_len = wav_len / (0.0125 * 24000) = wav_len / 300
        wav_id_lens = wav_lens // 300 # 300 = 0.0125 * 24000

        seqs = []
        seq_lens = []
        seq_sen_ids = []
        pos_ids = []
        max_seq_len = max([w+t for w, t in zip(wav_id_lens, text_lens)])
        for i in range(batch):
            wav_id, text_id, wav_len, text_len = \
                wav_ids[i], text_ids[i], wav_id_lens[i], text_lens[i]
            cur_seq_len = wav_len + text_len
            seq = torch.cat([
                torch.tensor([self.tokenizer.bos]).to(device),
                text_id[:text_len],
                torch.tensor([self.tokenizer.sep]).to(device),
                wav_id[:wav_len, 0],
                torch.tensor([self.tokenizer.eos]).to(device)
                ], dim=0)
            seq = F.pad(
                seq,
                (0, max_seq_len - cur_seq_len),
                mode="constant",
                value=self.tokenizer.pad,
            )
            seqs.append(seq)
            seq_lens.append(seq.shape[0])

             # 区分是text还是wav
            seq_sen_id = torch.tensor([1] * (text_len + 2) + [2] * (wav_len + 1))
            seq_sen_id = F.pad(
                seq_sen_id,
                (0, max_seq_len - cur_seq_len),
                mode="constant",
                value=self.tokenizer.pad,
            )
            seq_sen_ids.append(seq_sen_id)

            pos_id = torch.tensor(list(range(text_len + 2)) + list(range(wav_len + 1)))
            pos_id = F.pad(
                pos_id,
                (0, max_seq_len - cur_seq_len),
                mode="constant",
                value=self.tokenizer.pad,
            )
            pos_ids.append(pos_id)

        seqs = torch.stack(seqs, dim=0)
        seq_lens = torch.tensor(seq_lens).to(device)
        pos_ids = torch.stack(pos_ids, dim=0)
        seq_sen_ids = torch.stack(seq_sen_ids, dim=0)

        return seqs, seq_lens, pos_ids, seq_sen_ids

    @torch.no_grad()
    def get_codec_codes(self, x):
        output = self.requires["ss"](x)[2]
        output = torch.stack(output, dim=2) # [b, t, n_code]
        return output
