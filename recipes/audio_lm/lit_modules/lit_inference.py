import os
import re
import unicodedata

import pytorch_lightning as pl
import torch

from recipes.audio_lm.lit_modules.v4_1.lit_coarse_3ar import CoarseModule
from recipes.audio_lm.lit_modules.v4_1.lit_fine import FineModule
from recipes.audio_lm.lit_modules.v4_1.lit_semantic import SemanticModule
from samantha.utils.hparams import DotDict


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def save_wav(audio, output_file, sr=24000):
    from scipy.io.wavfile import write

    audio = audio * 32768.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return


class InferenceModule(pl.LightningModule):
    def __init__(
        self,
        semantic_ckpt,
        coarse_ckpt,
        fine_ckpt,
        out_dir,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.semantic_module = SemanticModule.load_from_checkpoint(semantic_ckpt).eval()
        self.coarse_module = CoarseModule.load_from_checkpoint(coarse_ckpt).eval()
        self.fine_module = FineModule.load_from_checkpoint(fine_ckpt).eval()
        # self.semantic_module = semantic_module.eval().freeze()
        # self.coarse_module = coarse_module.eval().freeze()
        # self.fine_module = fine_module.eval().freeze()
        self.out_dir = out_dir
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
            - torch.arange(self.extra_params.num_res, device=coarse_samples.device)
            * 1024
        )  # [b, t, n_codebook]
        vqgan_inputs = vqgan_inputs.transpose(
            1, 2
        )  # [b, t, n_codebook] -> [b, n_codebook, t]
        wavs = self.requires["ss_dec"](vqgan_inputs).squeeze(1)
        for i, wav in enumerate(wavs):
            wav_dir = os.path.join(self.out_dir, categories[i])
            os.makedirs(wav_dir, exist_ok=True)

            fp = os.path.join(wav_dir, f"{slugify(prompts[i])[:128]}")
            print(f"[Saving] {fp}")
            save_wav(wav.cpu().numpy(), fp + ".wav", sr=24000)
            with open(fp + ".txt", "w") as prompt_txt:
                prompt_txt.write(prompts[i])
