import os

import pytorch_lightning as pl
import torch
from bytedance import easycycle

import samantha.core.infer_dtype as infer
from samantha.utils.hparams import DotDict

from ...requires.model_initializer import (
    init_mulan,
    init_mulan_centers,
    init_sound_stream_decoder,
)
from ...utils.utils import dump_wav, save_wav, slugify
from ..v4_1.lit_coarse_3ar import CoarseModule
from ..v4_1.lit_fine import FineModule
from ..v4_1.lit_semantic import SemanticModule


class MusicLMInfer(pl.LightningModule):
    r"""This module aims to provide a unified wrapper for inference of MusicLM.
    User can and only can call interface ``predict_step``.

    This module accept checkpoint paths which should contain all needed modules'
    checkpoints. If any one is missing, it would raise an error.

    User only can specify checkpoint paths via easycycle api

    .. code-block:: shell

        # via environment variables
        export semantic=<semantic_ckpt_path>
        export coarse=<coarse_ckpt_path>
        ...
        python3 samantha.main predict --config <config-path> ...

    Attributes:
        required_modules (List[str]): defined all required modules

    Args:
        checkpoint_path (Optional[Dict[str, str]]): a dictionary should contain
            all required modules' checkpoint paths

    Raises:
        RuntimeError: raised when use this module not on stage ``predict``.
        ValueError: raised when any needed module's checkpoint is missing.

    """

    required_modules = [
        "mulan",
        "mulan_centers",
        "semantic",
        "coarse",
        "fine",
        "sound_stream_decoder",
    ]

    def __init__(self, extra_params=None):
        super().__init__()
        self.extra_params = DotDict(extra_params)
        print(self.extra_params)
        self.ckpts = self._parse_checkpoints()
        print(f"ckpts: {self.ckpts}")

    def setup(self, stage: str) -> None:
        if stage != "predict":
            raise RuntimeError("This module only for stage predict.")

        self.print("Loading semantic_model...")
        self.semantic_lit_module = SemanticModule.load_from_checkpoint(
            self.ckpts.semantic, map_location=self.device
        ).eval()
        self.print("Loading coarse_model...")
        self.coarse_lit_module = CoarseModule.load_from_checkpoint(
            self.ckpts.coarse, map_location=self.device
        ).eval()
        self.print("Loading fine_model...")
        self.fine_lit_module = FineModule.load_from_checkpoint(
            self.ckpts.fine, map_location=self.device
        ).eval()

        cache_dir = self.extra_params.get("cache_dir", ".cache")
        self.print("Loading sound stream decoder...")
        self.ss_dec = init_sound_stream_decoder(
            self.ckpts.sound_stream_decoder, self.local_rank, cache_dir=cache_dir
        )["ss_dec"]

        self.print("Loading mulan model...")
        self.mulan = init_mulan(
            self.ckpts.mulan,
            local_rank=self.local_rank,
            version=self.extra_params.get("mulan_version", "149"),
            cache_dir=cache_dir,
        )
        self.mulan_model = self.mulan.pop("mulan")
        self.mulan_centers = init_mulan_centers(
            self.ckpts.mulan_centers, self.local_rank, cache_dir
        )["mulan_centers"]

    def _parse_checkpoints(self):
        checkpoint_path = DotDict()
        if self.extra_params.get("from_cli", False):
            checkpoint_path.semantic = self.extra_params.get("AudioLM_semantic", None)
            checkpoint_path.coarse = self.extra_params.get("AudioLM_coarse", None)
            checkpoint_path.fine = self.extra_params.get("AudioLM_fine", None)
            checkpoint_path.sound_stream_decoder = self.extra_params.get(
                "SoundStream", None
            )
            checkpoint_path.mulan = self.extra_params.get("MuLan", None)
            checkpoint_path.mulan_centers = self.extra_params.get("mulan_centers", None)
        else:
            ckpts = easycycle.get_ckpt_groups()
            assert len(ckpts) == 1
            ckpts = ckpts[0].ckpts
            ckpts = {e.resource: e for e in ckpts}

            checkpoint_path.semantic = ckpts["AudioLM_semantic"].ckpt_hdfs
            checkpoint_path.coarse = ckpts["AudioLM_coarse"].ckpt_hdfs
            checkpoint_path.fine = ckpts["AudioLM_fine"].ckpt_hdfs
            checkpoint_path.sound_stream_decoder = ckpts["SoundStream"].ckpt_hdfs
            checkpoint_path.mulan = ckpts["MuLan"].ckpt_hdfs
            checkpoint_path.mulan_centers = self.extra_params.get("mulan_centers", None)

        for module in self.required_modules:
            if checkpoint_path.get(module, None) is None:
                raise ValueError(f"{module}'s checkpoint must be provided.")

        return checkpoint_path

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        r"""Perform music lm inference, typically it will execute as following order
        ``mulan_inference -> text2semantic -> semantic2coarse -> coarse2fine ->
        sound_stream_dec -> similarity_score``.

        Args:
            batch:
            batch_idx:
            dataloader_idx:

        Returns:
            List[infer.Container]: predict output

        """
        text_embs = []
        prompts = batch["text"]
        categories = batch["category"]
        for text in batch["text"]:
            text_emb = self.mulan["mulan_infer_fn"](
                self.mulan_model, text=text, device="cuda"
            )
            text_embs.append(text_emb)

        mulan_embeds = torch.cat(text_embs, dim=0)
        mulan_tokens, ds = self.mulan["mulan_rvq_fn"](mulan_embeds, self.mulan_centers)
        semantic_samples = self.semantic_lit_module.predict(mulan_tokens)
        coarse_samples = self.coarse_lit_module.predict(mulan_tokens, semantic_samples)
        fine_samples = self.fine_lit_module.predict(coarse_samples)

        bs = coarse_samples.size(0)
        coarse_samples = coarse_samples.view([bs, -1, self.extra_params.num_coarse])
        fine_samples = fine_samples.view([bs, -1, self.extra_params.num_fine])
        vqgan_inputs = (
            torch.cat([coarse_samples, fine_samples], dim=2)
            - torch.arange(self.extra_params.num_res, device=coarse_samples.device)
            * 1024
        )  # [b, t, n_codebook]
        # [b, t, n_codebook] -> [b, n_codebook, t]
        vqgan_inputs = vqgan_inputs.transpose(1, 2)
        wavs = self.ss_dec(vqgan_inputs).squeeze(1)
        ret = []
        output_wav_dir = self.extra_params.get("output_wav_dir", ".output_wav")
        for prompt, category, wav in zip(prompts, categories, wavs):
            os.makedirs(f"{output_wav_dir}/{category}", exist_ok=True)
            save_wav(
                wav.cpu().numpy(),
                f"{output_wav_dir}/{category}/{slugify(prompt)[:128]}.wav",
            )
            dumped_wav = dump_wav(wav.cpu().numpy(), sr=24000)
            ret.append(
                infer.Container(
                    input=infer.Element(infer.Dtype.TEXT, slugify(prompt)),
                    output=infer.Element(infer.Dtype.AUDIO, dumped_wav),
                    meta={"category": category},
                )
            )
        return ret
