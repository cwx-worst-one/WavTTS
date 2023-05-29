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
            * self.extra_params.soundstream_codebook_size
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


class GTInferenceModule(pl.LightningModule):
    def __init__(
        self,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.gt_infer_mode = self.extra_params.gt_infer_mode
        if self.gt_infer_mode in ["mulan->semantic->coarse->fine", "semantic->coarse->fine", "coarse->fine"]:
            self.fine_module = FineModule.load_from_checkpoint(self.extra_params.fine_ckpt).eval()
            if self.gt_infer_mode in ["mulan->semantic->coarse->fine", "semantic->coarse->fine"]:
                self.coarse_module = CoarseModule.load_from_checkpoint(self.extra_params.coarse_ckpt).eval()
                if self.gt_infer_mode in ["mulan->semantic->coarse->fine"]:
                    self.semantic_module = SemanticModule.load_from_checkpoint(self.extra_params.semantic_ckpt).eval()
        else:
            raise ValueError(f"Invalid ground truth inference mode: {self.gt_infer_mode}")
        self.requires = {}
        self.load_required_modules()
    
    def _load_required_module(self, name):
        hpath, initializer = self.hparams.required_modules[name]
        self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def load_required_modules(self):
        if self.gt_infer_mode in ["mulan->semantic->coarse->fine", "semantic->coarse->fine", "coarse->fine"]:
            self._load_required_module("soundstream_dec")
            if self.gt_infer_mode in ["mulan->semantic->coarse->fine"]:
                self._load_required_module("mulan")
                self._load_required_module("mulan_centers")
            elif self.gt_infer_mode in ["semantic->coarse->fine"]:
                self._load_required_module("wav2vec")
                self._load_required_module("semantic_centers")
            elif self.gt_infer_mode in ["coarse->fine"]:
                self._load_required_module("soundstream")

    @torch.no_grad()
    def get_mulan_tokens(self, x):
        mulan_embeds = self.requires["mulan_infer_fn"](
            model=self.requires["mulan"], music=x.float(), device=x.device
        )
        mulan_tokens, _ = self.requires["mulan_rvq_fn"](
            mulan_embeds, self.requires["mulan_centers"]
        )
        return mulan_tokens

    @torch.no_grad()
    def get_soundstream_tokens(self, x):
        output = self.requires["ss"](x.float())[2]
        output = torch.stack(output, dim=2)
        return output

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

    def _predict_step(self, batch, batch_idx, round):

        if isinstance(batch, list):
            batch = batch[0]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        
        device = batch.device
        bs = batch.size(0)

        if self.extra_params.gt_infer_mode == "mulan->semantic->coarse->fine":
            mulan_ids = self.get_mulan_tokens(batch)
            semantic_tokens = self.semantic_module.predict(mulan_ids, self.extra_params)
            coarse_samples = self.coarse_module.predict(semantic_tokens, self.extra_params)
        elif self.extra_params.gt_infer_mode == "semantic->coarse->fine":
            semantic_tokens = self.get_wav2vec_tokens(batch)
            coarse_samples = self.coarse_module.predict(semantic_tokens, self.extra_params)
        elif self.extra_params.gt_infer_mode == "coarse->fine":
            soundstream_ids = self.get_soundstream_tokens(batch)
            coarse_samples = (
                soundstream_ids[:, :, 0 : self.extra_params.num_coarse]
                + torch.arange(self.extra_params.num_coarse, device=device) * self.extra_params.soundstream_codebook_size
            ).reshape((bs, -1))

        fine_samples = self.fine_module.predict(coarse_samples, self.extra_params)

        coarse_samples = coarse_samples.view([bs, -1, self.extra_params.num_coarse])
        fine_samples = fine_samples.view([bs, -1, self.extra_params.num_fine])
        vqgan_inputs = (
            torch.cat([coarse_samples, fine_samples], dim=2)
            - torch.arange(self.extra_params.num_coarse + self.extra_params.num_fine, device=coarse_samples.device)
            * self.extra_params.soundstream_codebook_size
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
            save_wav(wav.cpu().numpy(), f"{fp}.{round}.wav", sr=24000)
            save_wav(batch[i].cpu().numpy(), f"{fp}.wav", sr=24000)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        for i in range(self.extra_params.num_rounds):
            self._predict_step(batch, batch_idx, i)
