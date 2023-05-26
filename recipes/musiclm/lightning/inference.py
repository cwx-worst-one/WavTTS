import os

import pytorch_lightning as pl
import torch

from recipes.musiclm.lightning.modules import SemanticModule
from recipes.musiclm.lightning.modules import CoarseModule
from recipes.musiclm.lightning.modules import FineModule
from samantha.utils.hparams import DotDict
from ..inference.utils import slugify, save_wav
from recipes.musiclm.models.compat.semantic_model import w2v_bert_tokenization


class InferenceModule(pl.LightningModule):
    def __init__(
        self,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.semantic_module = SemanticModule.load_from_checkpoint(self.extra_params.semantic_ckpt).eval()
        self.coarse_module = CoarseModule.load_from_checkpoint(self.extra_params.coarse_ckpt).eval()
        self.fine_module = FineModule.load_from_checkpoint(self.extra_params.fine_ckpt).eval()
        self.requires = {}
        self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))
            if name == "mulan_centers":
                assert self.extra_params.mulan_num_rvq == self.requires["mulan_centers"].shape[0]
    
    def _predict_step(self, batch, round):
        text_embs = []
        prompts = batch["text"]
        categories = batch["category"]
        for text in batch["text"]:
            text_emb = self.requires["mulan_infer_fn"](
                self.requires["mulan"], text=text, device="cuda"
            )
            text_embs.append(text_emb)

        mulan_embeds = torch.cat(text_embs, dim=0)
        mulan_ids, ds = self.requires["mulan_rvq_fn"](
            mulan_embeds, self.requires["mulan_centers"]
        )
        bs = mulan_ids.size(0)
        semantic_samples = self.semantic_module.predict(mulan_ids, self.extra_params)
        coarse_samples = self.coarse_module.predict(semantic_samples, self.extra_params)
        fine_samples = self.fine_module.predict(coarse_samples, self.extra_params)
        
        coarse_samples = coarse_samples.view([bs, -1, self.extra_params.num_coarse])
        fine_samples = fine_samples.view([bs, -1, self.extra_params.num_fine])
        vqgan_inputs = (
            torch.cat([coarse_samples, fine_samples], dim=2)
            - torch.arange(self.extra_params.num_coarse + self.extra_params.num_fine, device=coarse_samples.device)
            * 1024
        )  # [b, t, n_codebook]
        vqgan_inputs = vqgan_inputs.transpose(
            1, 2
        )  # [b, t, n_codebook] -> [b, n_codebook, t]
        wavs = self.requires["ss_dec"](vqgan_inputs).squeeze(1)
        for i, wav in enumerate(wavs):
            wav_dir = os.path.join(self.extra_params.output_dir, categories[i])
            os.makedirs(wav_dir, exist_ok=True)
            fp = os.path.join(wav_dir, f"{slugify(prompts[i])[:128]}.{round}.wav")
            print(f"[Saving] {fp}")
            save_wav(wav.cpu().numpy(), fp, sr=24000)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        for i in range(self.extra_params.num_rounds):
            self._predict_step(batch, i)


class SemanticGTInferenceModule(pl.LightningModule):
    def __init__(
        self,
        coarse_ckpt,
        fine_ckpt,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.coarse_module = CoarseModule.load_from_checkpoint(coarse_ckpt).eval()
        self.fine_module = FineModule.load_from_checkpoint(fine_ckpt).eval()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @torch.no_grad()
    def get_wav2vec_tokens(self, x):
        wav2vec_tokens = w2v_bert_tokenization(
            frontend=self.requires["ssl_frontend"],
            w2v_model=self.requires["semantic"],
            wavs=x.float(),
            centers=self.requires["semantic_centers"],
            device=x.device,
        )
        return wav2vec_tokens

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, list):
            batch = batch[0]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        if self.extra_params.cross_attn:
            if self.extra_params.mert:
                semantic_tokens = self.get_mert_embeds(batch)
            else:
                semantic_tokens = self.get_wav2vec_embeds(batch)
        else:
            semantic_tokens = self.get_wav2vec_tokens(batch)
        coarse_samples = self.coarse_module.predict(semantic_tokens, self.extra_params)
        fine_samples = self.fine_module.predict(coarse_samples)

        bs = coarse_samples.size(0)
        coarse_samples = coarse_samples.view([bs, -1, self.extra_params.num_coarse])
        fine_samples = fine_samples.view([bs, -1, self.extra_params.num_fine])
        vqgan_inputs = (
            torch.cat([coarse_samples, fine_samples], dim=2)
            - torch.arange(self.extra_params.num_coarse + self.extra_params.num_fine, device=coarse_samples.device)
            * 1024
        )  # [b, t, n_codebook]
        vqgan_inputs = vqgan_inputs.transpose(
            1, 2
        )  # [b, t, n_codebook] -> [b, n_codebook, t]
        wavs = self.requires["ss_dec"](vqgan_inputs).squeeze(1)
        for i, wav in enumerate(wavs):
            wav_dir = os.path.join(self.extra_params.output_dir)
            os.makedirs(wav_dir, exist_ok=True)
            fp = os.path.join(wav_dir, f"{batch_idx * bs + i}")
            print(f"[Saving] {fp}")
            save_wav(wav.cpu().numpy(), fp + f".{self.current_epoch}.wav", sr=24000)
            save_wav(batch[i].cpu().numpy(), fp + ".wav", sr=24000)
