import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np

from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm
import math

from samantha.utils.hparams import DotDict
from recipes.bark.lit_modules.sample import sample
from s3a.providers.ctiga.utils.generation import InferenceParams
from samantha.utils.model_metric import ModelMetric

def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask

class VAET2SLangSpkModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        logits_criterion_cls,
        dense_criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=True,
        stop_token_loss_weight=1.0,
        use_lang_id=False,
        use_spk_id=False,
        resume_ckpt_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.logits_criterion = logits_criterion_cls()
        self.dense_criterion = dense_criterion_cls()
        self.requires = {}
        self.use_lang_id = use_lang_id
        print("use_lang_id: ", self.use_lang_id)
        self.use_spk_id = use_spk_id
        print("use_spk_id: ", self.use_spk_id)

        if checkpointing:
            self.model.gradient_checkpointing_enable()
        self.stop_token_loss_weight = stop_token_loss_weight

        resume_ckpt_path = None
        print("resume_ckpt_path: ", resume_ckpt_path)
        if resume_ckpt_path:
            state_dict = torch.load(resume_ckpt_path, map_location=torch.device('cpu'))['state_dict']
            new_state_dict = []
            new_state_dict = {k.replace("model.",""):v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)
            print("Loading state_dict from {} successfully".format(resume_ckpt_path))

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
                frontend_inputs = {
                        "phone": batch["phone"],
                        "tone": batch["tone"],
                        }

                bns, stop_tokens = batch["bn"], batch["stop_token"]
                text_lens, bn_lens = batch["text_lens"], batch["bn_lens"]

                seq_lens = text_lens + bn_lens
                seq_len = max(seq_lens)

                loss_mask = sequence_mask(seq_lens, device="cuda")
                text_loss_mask = sequence_mask(text_lens, device="cuda")
                text_loss_mask = F.pad(text_loss_mask, (0, seq_len - text_loss_mask.shape[1]), "constant", 0)
                z_loss_mask = loss_mask - text_loss_mask

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            ret_dict = self.model(frontend_inputs, bns, text_lens, bn_lens, lang_seqs=batch["lang_seq"], spk_seqs=batch["spk_seq"])
            pred_stop_token = ret_dict["stop_token"]
            pred_dense = ret_dict["dense"]

        pred_stop_token = pred_stop_token[:, 0:seq_len - 1, :]
        pred_dense = pred_dense[:, 0:seq_len - 1, :]

        bsz, bn_t, bn_c = bns.shape
        targets_dense = []
        for i in range(bsz):
            targets_dense.append(
                F.pad(bns[i, :bn_lens[i], :], (0, 0, text_lens[i], seq_len-(bn_lens[i]+text_lens[i])), "constant", 0)
            )
        targets_dense = torch.stack(targets_dense)[:, 1:seq_len, :]

        target_m, target_logs = torch.split(targets_dense, bn_c//2, dim=-1)
        pred_m, pred_logs = torch.split(pred_dense, bn_c//2, dim=-1)

        # kl_loss
        kl_loss = self.dense_criterion(pred_m, pred_logs, 
            target_m.detach(), target_logs.detach(), z_mask=z_loss_mask[:, 1:])

        # stop_token_loss
        target_stop_token = stop_tokens[:, 1:seq_len]
        stop_token_loss = self.logits_criterion(pred_stop_token.float(), target_stop_token, mask=z_loss_mask[:, 1:])
        stop_token_accu = ((pred_stop_token.argmax(dim=-1) == target_stop_token).float() * z_loss_mask[:, 1:]).sum() / z_loss_mask[:, 1:].sum() * 100

        total_loss = kl_loss + stop_token_loss * self.stop_token_loss_weight

        batch_tokens= bsz * seq_len

        self.log_dict(
            {
                "kl_loss": kl_loss.item(),
                "stop_token_loss": stop_token_loss.item(),
                "stop_token_accu": stop_token_accu.item(),
                "loss": total_loss.item(),
                "bsz": bsz,
                "seqlen": seq_len,
                "batch_tokens": batch_tokens,
            },
            prog_bar=True,
            sync_dist=True
        )

        self.model_metric.update(
            num_tokens=batch_tokens,
            stage=self.trainer.state.stage,
            model_kwargs=dict(batch_size=bsz, seqlen=seq_len),
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
        bns = batch["bn"]
        text_lens, bn_lens = batch["text_lens"], batch["bn_lens"]
        lang_seq, infer_lang_id = batch["lang_seq"], batch["infer_lang_id"]
        spk_seq, infer_spk_id = batch["spk_seq"], batch["infer_spk_id"]

        frontend_inputs = {
                "phone": batch["phone"],
                "tone": batch["tone"],
                }

        b = bns.shape[0]
        
        # llama style inference
        self.model.params.use_cache = True
        start_pos = 0
        
        inference_params = InferenceParams(
            max_sequence_len=8000, 
            max_batch_size=b,
            fused_ft_kernel=False
        )
        semantic_outputs = []
        z_list = []

        max_step = text_lens[0] * 10 - bn_lens[0]
        
        with torch.autocast(device_type="cuda", enabled=True):
            for i in tqdm(range(max_step)):
                if i == 0:
                    model_outputs = self.model(frontend_inputs, bns, text_lens, bn_lens, start_pos=start_pos, inference_params=inference_params, lang_seqs=lang_seq, spk_seqs=spk_seq)
                    input_len = text_lens[0] + bn_lens[0]
                else:
                    model_outputs = self.model(frontend_inputs, bns, text_lens, bn_lens, start_pos=start_pos, use_cache=True, inference_params=inference_params, lang_seqs=lang_seq, spk_seqs=spk_seq)
                    input_len = 1
                    z_list.append(model_outputs['bn_in_z'])

                pred_stop_token = model_outputs["stop_token"][:, -1, :]
                samples = torch.argmax(pred_stop_token)
                if i > 10 and samples.item() == 1:
                    break

                pred_dense = model_outputs["dense"][:, -1:, :]

                start_pos += input_len
                inference_params.sequence_len_offset = start_pos

                # next infer
                semantic_outputs.append(pred_dense)
                bns = pred_dense
                bn_lens[0] = 1

                lang_id = torch.from_numpy(np.asarray([infer_lang_id]))
                lang_id = lang_id.unsqueeze(0)
                lang_id = lang_id.to(pred_dense.device)
                lang_seq = lang_id

                spk_seq = None
                if self.use_spk_id:
                    spk_id = torch.from_numpy(np.asarray([infer_spk_id]))
                    spk_id = spk_id.unsqueeze(0)
                    spk_id = spk_id.to(pred_dense.device)
                    spk_seq = spk_id

        z_outputs = torch.cat(z_list, dim=1) # [b, t, c]
        semantic_outputs = torch.cat(semantic_outputs, dim=1) # [b, t, c]
        self.model.params.use_cache = False
        return z_outputs, semantic_outputs

    predict = inference_from_text




