import pytorch_lightning as pl
import torch
import numpy as np

from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict


class CoarseModule(pl.LightningModule):

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

        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                wavs_16k, wavs_24k, texts, wav_16k_lens, wav_24k_lens, text_lens = batch
                wavs_16k, wavs_24k = wavs_16k.float() / 32768.0, wavs_24k.float() / 32768.0
                input_tokens, loss_mask = self.prepare_feature(wavs_16k, wavs_24k, texts, wav_16k_lens, wav_24k_lens, text_lens)

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            logits = self.model(input_tokens)["logits"]

        if self.local_rank == 0:
            print("Bark: Coarse Training, V1...")

        x = logits[:, 0:-1, :]
        targets = input_tokens[:, 1:]
        loss = self.criterion(x.float(), targets, mask=loss_mask[:, 1:])
        accu = ((x.argmax(dim=-1) == targets).float() * loss_mask[:, 1:]).sum() / loss_mask[:, 1:].sum() * 100
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
        device = wavs_16k.device
        b = wavs_16k.size(0)

        # semantic 255
        semantic_codes, semantic_length = self.get_semantic_codes(wavs_16k, wav_16k_lens)
        new_semantic_codes = torch.zeros(size=[b, 256], device=device, dtype=semantic_codes.dtype) + self.n_semantic # eos
        beg_ends = []
        for i, l in enumerate(semantic_length):
            # random slice
            if l < 256:
                new_semantic_codes[i, 0:l] = semantic_codes[i, 0:l]
                beg_ends.append([0, l.item()])
            else:
                beg = torch.randint(low=0, high=l - 255 + 1, size=[]).item()
                new_semantic_codes[i, 0:255] = semantic_codes[i, beg:beg + 255]
                beg_ends.append([beg, beg + 255])

        # coarse 255*2*1.6 = 816
        codec_codes = self.get_codec_codes(wavs_24k)[:, :, 0:2] + torch.arange(2, device=device) * 1024
        b, t, _ = codec_codes.size()
        codec_codes = codec_codes.reshape([b, t*2])
        # eos
        new_codec_codes = torch.zeros(size=[b, 816], device=device, dtype=codec_codes.dtype) + 2 * 1024 + 1
        codec_lens = torch.zeros(size=[b,], device=device, dtype=torch.long)

        for i, beg_ends in enumerate(beg_ends):
            beg, end = beg_ends
            beg = int(np.ceil(beg * 2 * 1.6))
            end = min(int(np.ceil(end * 2 * 1.6)), codec_codes.size(1))
            new_codec_codes[i, 0:end - beg] = codec_codes[i, beg:end]
            codec_lens[i] = end - beg
        new_codec_codes = torch.nn.functional.pad(new_codec_codes, (1, 0))
        # bos
        new_codec_codes[:, 0] = 2 * 1024
        # offset: smantic 8192 + semantic eos 1
        new_codec_codes += self.n_semantic + 1

        input_tokens = torch.cat([new_semantic_codes, new_codec_codes], dim=1)
        # loss mask
        semantic_mask = torch.arange(input_tokens.size(1), device=device).unsqueeze(0) >= 256
        semantic_mask = semantic_mask.repeat(b, 1)
        coarse_mask = torch.arange(input_tokens.size(1), device=device).unsqueeze(0) < (codec_lens.unsqueeze(1) + 256 + 2)
        loss_mask = torch.logical_and(semantic_mask, coarse_mask)
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
