import pytorch_lightning as pl
import torch
import numpy as np

from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict


class AcousticModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=True,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
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

        with self.profiler.profile("[LightningModule]AcousticModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                wavs_16k, wavs_24k, texts, wav_16k_lens, wav_24k_lens, text_lens = batch
                wavs_24k = wavs_24k.float() / 32768.0
                wavs_16k = wavs_16k.float() / 32768.0
                codec_codes, codec_lens, semantic_codes, semantic_length = self.prepare_feature(wavs_16k, wavs_24k, texts, wav_16k_lens, wav_24k_lens, text_lens)

        with self.profiler.profile("[LightningModule]AcousticModule.model_forward"):
            model_outputs = self.model(codec_codes, codec_lens, semantic_codes, semantic_length)

        if self.local_rank == 0:
            print("Bark: SoundStorm NAR Training, V1...")

        targets = model_outputs["targets"]
        logits = model_outputs["logits"]
        loss_mask = model_outputs["loss_mask"]
        loss = self.criterion(logits.float(), targets, mask=loss_mask)
        accu = ((logits.argmax(dim=-1) == targets).float() * loss_mask).sum() / loss_mask.sum().clamp(1.0) * 100
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
        semantic_codes, semantic_length = self.get_semantic_codes(wavs_16k, wav_16k_lens)
        return codec_codes, codec_lens, semantic_codes, semantic_length

    @torch.no_grad()
    def get_semantic_codes(self, wavs_16k, wav_16k_lens):
        infer_fn = self.requires["semantic_infer_fn"]
        semantic_codes, semantic_length = infer_fn(
            self.requires["semantic"],
            self.requires["centroids"],
            wavs_16k,
            wav_16k_lens,
            wavs_16k.device,
        )
        return semantic_codes, semantic_length

    @torch.no_grad()
    def get_codec_codes(self, x):
        output = self.requires["ss"](x)[2]
        output = torch.stack(output, dim=2) # [b, t, n_code]
        return output

    @torch.no_grad()
    def inference_from_semantic(self, semantic, semantic_lens, prompt, prompt_len, strategy="min_entropy"):
        bsz, ts = semantic.size()
        bsz, tp, n_code = prompt.size()
        assert bsz == 1
        codec_lens = (semantic_lens * 1.6).ceil().long()
        prompt = torch.nn.functional.pad(prompt, (0, 0, 0, codec_lens.max() - tp))
        inputs = prompt
        # [1, t] < [b, 1]
        prompt_mask = torch.arange(prompt.size(1), device=device).unsqueeze(0) < prompt_lens.unsqueeze(1)
        inference_iters = [16, 8, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        for index, infer_iter in enumerate(inference_iters):
            layer_index = torch.LongTensor([index,]).to(device)
            cosine_mask = torch.zeros_like(inputs[:, :, 0]).bool()
            print("layer_index: ", layer_index)
            if infer_iter >= 1:
                temp = 0.7 / (2 ** index)
                pred_lens = ((codec_lens - prompt_lens) * (1 / infer_iter)).floor().long() # [b,]
                pred_lens = torch.stack([pred_lens] * infer_iter, dim=1) # [b, n]
                pred_lens[:, -1] = codec_lens - prompt_lens - pred_lens[:, 0:-1].sum(dim=1) # [b]
                for k in range(infer_iter):
                    model_outputs = model.model(inputs, codec_lens, semantic, semantic_lens, layer_index, prompt_mask, cosine_mask)
                    logits = model_outputs['logits'] # [b, t, d]
                    loss_mask = model_outputs['loss_mask'].bool() # [b, t]
                    # p_max决定了哪个位置被sample到，indices决定了被sample到的index
                    p = logits.softmax(dim=-1) # [b, t, d]
                    if strategy == 'max_probs':
                        p_max, indices = p.max(dim=2) # [b, t]
                    elif strategy == 'min_entropy':
                        p_max = (p * p.log()).sum(dim=2) # [b, t]
                        indices = logits.argmax(dim=2)
                        if index == len(inference_iters) - 1:
                            indices = logits.argmax(dim=2)
                        else:
                            indices = gumbel_sample(logits, temp=temp)

                    p_max = torch.where(loss_mask, p_max, torch.zeros_like(p_max) - 1e5)

                    # TODO: parallelize this implementation
                    k_values, k_indices = torch.topk(p_max, k=pred_lens[0, k], dim=1) # [b, k]
                    select_mask = torch.zeros_like(prompt_mask).bool()
                    for b in range(bsz):
                        for ind in k_indices[b]:
                            select_mask[b, ind] = True

                    select_mask = torch.logical_and(select_mask, loss_mask)
                    indices = torch.where(select_mask, indices, torch.zeros_like(indices))
                    inputs[:, :, index] = torch.where(select_mask, indices, inputs[:, :, index])
                    cosine_mask = torch.logical_or(cosine_mask, select_mask) # update cosine mask
            else:
                raise Exception