import os

import pytorch_lightning as pl
import torch

# from recipes.musiclm.lightning.modules import (
#     SemanticModule,
#     CoarseModule,
#     FineModule,
# )
from recipes.audio_lm.lit_modules.v4_1.lit_coarse_3ar import CoarseModule
from recipes.audio_lm.lit_modules.v4_1.lit_fine import FineModule
from recipes.audio_lm.lit_modules.v4_1.lit_semantic import SemanticModule
from samantha.utils.hparams import DotDict
from ..inference.utils import slugify, save_wav


class InferenceModule(pl.LightningModule):
    def __init__(
        self,
        semantic_ckpt,
        coarse_ckpt,
        fine_ckpt,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.semantic_module = SemanticModule.load_from_checkpoint(semantic_ckpt).eval()
        self.coarse_module = CoarseModule.load_from_checkpoint(coarse_ckpt).eval()
        self.fine_module = FineModule.load_from_checkpoint(fine_ckpt).eval()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        text_embs = []
        prompts = batch["text"]
        categories = batch["category"]
        for text in batch["text"]:
            text_emb = self.requires["mulan_infer_fn"](
                self.requires["mulan"], text=text, device="cuda"
            )
            text_embs.append(text_emb)

        mulan_embeds = torch.cat(text_embs, dim=0)
        mulan_tokens, ds = self.requires["mulan_rvq_fn"](
            mulan_embeds, self.requires["mulan_centers"]
        )
        semantic_samples = self.semantic_module.predict(mulan_tokens)
        coarse_samples = self.coarse_module.predict(mulan_tokens, semantic_samples)
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
            wav_dir = os.path.join(self.extra_params.output_dir, categories[i])
            os.makedirs(wav_dir, exist_ok=True)
            fp = os.path.join(wav_dir, f"{slugify(prompts[i])[:128]}")
            print(f"[Saving] {fp}")
            save_wav(wav.cpu().numpy(), fp + ".wav", sr=24000)
            with open(fp + ".txt", "w") as prompt_txt:
                prompt_txt.write(prompts[i])
