import pytorch_lightning as pl
import torch
import numpy as np

from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict


class ValleFineModule(pl.LightningModule):

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
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.tokenizer_len = tokenizer_len
        self.n_semantic = n_semantic
        self.extra_params = DotDict(extra_params)
        self.requires = {}

        torch._C._jit_set_bailout_depth(0)

        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):

        with self.profiler.profile("[LightningModule]FineModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                _, seq_lens, pos_ids, seq_sen_ids, full_seqs, _ = batch
 
        with self.profiler.profile("[LightningModule]FineModule.model_forward"):
            model_outputs = self.model(full_seqs, seq_lens, seq_sen_ids)

        if self.local_rank == 0:
            print("VALLE: Fine Training.")

        targets = model_outputs["targets"]
        targets = torch.where(model_outputs["loss_mask"], targets, torch.zeros_like(targets))
        logits = model_outputs["logits"]
        loss_mask = model_outputs["loss_mask"]
        loss = self.criterion(logits.float(), targets, mask=loss_mask)
        accu = ((logits.argmax(dim=-1) == targets).float() * loss_mask).sum() / loss_mask.sum() * 100
        self.log_dict(
            {
                "tr_loss": loss.item(),
                "accu": accu.item()
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
    def prepare_feature(self, wavs_16k, wavs_24k, texts, wav_16k_lens, wav_24k_lens, text_lens):
        device = wavs_24k.device
        b = wavs_24k.size(0)
        codec_codes = self.get_codec_codes(wavs_24k) # [b, t, n_code]
        codec_lens = torch.floor(wav_24k_lens / self.extra_params.hop_size).long()
        new_codec_codes = torch.zeros(size=[b, 408, codec_codes.size(-1)], device=device, dtype=torch.long)

        # random slice
        beg_ends = []
        for i, l in enumerate(codec_lens):
            l = l.item()
            if l <= 408:
                new_codec_codes[i, 0:l] = codec_codes[i, 0:l]
                beg_ends.append([0, l])
            else:
                beg = torch.randint(low=0, high=l - 408 + 1, size=[]).item()
                new_codec_codes[i, 0:408] = codec_codes[i, beg:beg + 408]
                beg_ends.append([beg, beg + 408])
        # sequence length
        codec_lens = torch.zeros(size=[b,], device=device, dtype=torch.long)
        for i, (beg, end) in enumerate(beg_ends):
            codec_lens[i] = end - beg
        return new_codec_codes, codec_lens

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
    def inference_from_coarse(self, coarse_tokens, unmask_len, seq_sen_id, mask2, tokenizer):
        b, t, n_code = coarse_tokens.size() # [b, t, n_code]
        device = coarse_tokens.device

        codec_tokens = coarse_tokens

        for i in range(1, 6):
            layer_index = torch.zeros(size=[b,], device=device, dtype=torch.long) + i
            model_outputs = self.model.predict(codec_tokens, unmask_len, seq_sen_id, layer_index)
            # sampling
            logits = model_outputs['logits'] # [B, T, 1026]
            logits[:, :, -2:] = -1e5
            samples = logits.argmax(dim=-1) + tokenizer.phone_token_num + 1
            # update codebook
            masked_samples = torch.where(mask2, codec_tokens[:, :, i], samples)

            codec_tokens[:, :, i] = masked_samples

        return codec_tokens
