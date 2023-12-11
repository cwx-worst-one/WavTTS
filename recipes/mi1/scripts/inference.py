import torchaudio
import os
import torch
from typing import Dict
from argparse import ArgumentParser
from hyperpyyaml import load_hyperpyyaml
from samantha.utils.hdfs_tools import hdfs_open, hdfs_get_cache
from samantha.utils.hparams import DotDict

from recipes.mi1.models.mi1 import MI1Input, MI1
from samantha.utils.logger import RankedLogger
from samantha.models.base import LightningModuleBase

logger = RankedLogger(__name__)


def load_config(hparams_fp: str):
    if hparams_fp.startswith("hdfs"):
        with hdfs_open(hparams_fp, "r") as fin:
            hparams = load_hyperpyyaml(fin)
    else:
        with open(hparams_fp, "r", encoding="utf-8") as fin:
            hparams = load_hyperpyyaml(fin)
    return DotDict(hparams)


from recipes.diffusion.models.diffusion_model.utils import init_diffusion
from recipes.diffusion.models.vocoder_model.utils import init_vocoder
from recipes.diffusion.models.diffusion_model.utils import run_diffusion
from recipes.diffusion.inference_from_text_lyrics2song import init_sampler
from samantha.dataio.data_bucket import data_bucket

class DiffusionModels(LightningModuleBase):
    def __init__(
        self,
        duration: int = 30,
        num_chunks: int = 1,
        diffusion_steps: float = 25,
        schedule_slope: float = 2.5,
        guidance_scale: float = 2.5,
        bf16_portion: float = 0.0,
        rank: int = 0,
        root_cache_dir: str = ".cache",
    ):
        super().__init__()
        self._num_chunks = num_chunks
        self._diffusion_steps = diffusion_steps
        self._schedule_slope = schedule_slope
        self._guidance_scale = guidance_scale
        self._bf16_portion = bf16_portion
        self._root_cache_dir = root_cache_dir

        self._diffusion_hpath = data_bucket(
            "models/diffusion/model_14_30s_finetune/checkpoints/last-minimal.ckpt"
        )
        self.diffusion_model = init_diffusion(
            self._diffusion_hpath,
            local_rank=rank,
            cache_dir=self.cache_dir,
            is_zh_token=False,
        )["diffusion"]

        self.sampler = init_sampler(
            checkpoint_path=None,
            local_rank=rank,
            cache_dir=self.cache_dir,
            duration=duration,
            sequence_length=None,
        )["sampler"]

        self._vocoder_hpath = data_bucket(
            "models/diffusion/assets/soundstream-step=374999-val_sdr=12.9557-minimal.ckpt"
        )
        self.vocoder_model = init_vocoder(
            self._vocoder_hpath, local_rank=rank, cache_dir=self.cache_dir
        )["vocoder"]


        self.diffusion_model = self.diffusion_model.eval()
        self.sampler = self.sampler.eval()
        self.vocoder_model = self.vocoder_model.eval()



    @property
    def params(self) -> Dict[str, float]:
        return dict(
            num_chunks=self._num_chunks,
            diffusion_steps=self._diffusion_steps,
            schedule_slope=self._schedule_slope,
            guidance_scale=self._guidance_scale,
            bf16_portion=self._bf16_portion,
        )

    @property
    def requires_dict(self) -> dict:
        return {
            "diffusion": self.diffusion_model,
            "sampler": self.sampler,
            "vocoder": self.vocoder_model,
        }

    @property
    def cache_dir(self):
        return os.path.join(self._root_cache_dir, "diffusion")

    def tokens_to_audio(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        return run_diffusion(
            requires=self.requires_dict, samples=audio_tokens, params=self.params
        )


from recipes.bigmusic.lightning.semantic_modules import SemanticModule
from recipes.bigmusic.lightning.semantic_modules import (
    process_eos_indexes,
    truncate_wav_to_eos,
)


class Q4SemanticModel(LightningModuleBase):
    def __init__(
        self,
        sample_rate: int = 24000,
        duration: int = 30,
        semantic_temperature: float = 1.0,
        sample_mode: str = "top_p",
        sample_thresh: float = 0.95,
        beam_size: int = 1,
    ):
        super().__init__()
        self._sample_rate = sample_rate
        self._duration = duration
        self._semantic_temperature = semantic_temperature
        self._sample_mode = sample_mode
        self._sample_thresh = sample_thresh
        self._beam_size = beam_size  # set to 4 if using reranker, 1 without reranker

        self._semantic_hpath = data_bucket(
            "models/baseline_models/q4/20231102-new-baseline-mixed-ctiga-196k/checkpoints/step=196000-tr_loss=3.9746-val_accu_0=24.90.ckpt",
            cache=True,
        )
        self.semantic_module: SemanticModule = SemanticModule.load_from_checkpoint(
            self._semantic_hpath
        )

        self.semantic_module.hparams.required_modules["mulan"]["hpath"] = data_bucket(
            "models/mulan/ongoing/mulan-step=024600-median_rank_1=110-kaggle-minimal.ckpt",
            cache=True,
        )
        self.semantic_module.load_required_modules(
            ignore=(
                "bestrq",
                "sampler",
                "diffusion",
                "vocoder",
                "chord",
                "chord_lms",
                "structure",
                "asr",
            )
        )
        self.semantic_module.model.to(torch.float16)

    @property
    def extra_params(self) -> DotDict:
        return DotDict(
            sample_rate=self._sample_rate,
            duration=self._duration,
            semantic_temperature=self._semantic_temperature,
            sample_mode=self._sample_mode,
            sample_thresh=self._sample_thresh,
            beam_size=self._beam_size,
        )

    def generate(self, batch) -> torch.Tensor:
        semantic_samples = self.semantic_module.predict(
            batch, self.extra_params, beam=self.extra_params.beam_size
        )
        semantic_samples, eos_index_list = process_eos_indexes(
            semantic_samples, self.semantic_module, self.extra_params.sample_rate
        )
        return semantic_samples


class Inference(LightningModuleBase):
    def __init__(self, m1_model: MI1):
        super().__init__()
        self.diffusion_models = DiffusionModels()
        # self.semantic_model = Q4SemanticModel()
        # self.sample_rate = 24000
        
        # M1 is trained in bfloat16 precision
        self.m1_model = m1_model.to(torch.bfloat16)
        self.m1_model = self.m1_model.eval()


    @property
    def sample_rate(self):
        return self.m1_model.config.sample_rate

    def generate(self, batch):
        # audio_tokens = self.semantic_model.generate(batch)

        lyrics = batch["lyrics"]
        inputs = MI1Input(lyrics=lyrics)
        audio_tokens = self.m1_model.generate(inputs)

        eos_padding_id = 0 # TODO why?
        eos_mask = torch.cumsum(audio_tokens == self.m1_model.eos_token_id, 1) > 0
        audio_tokens[eos_mask] = eos_padding_id
        audio = self.diffusion_models.tokens_to_audio(audio_tokens)
        return audio


from recipes.bigmusic.datasets.inference import inference_dataset_from_prompt
from recipes.bigmusic.datasets.lyrics import LyricsDataModule

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--m1_ckpt_path", type=str)
    parser.add_argument("--prompt_path", type=str)

    parser.add_argument("--merge_url2index", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.conf)

    # prepare M1 model:
    m1_model: MI1 = cfg.pl_module
    if args.m1_ckpt_path:
        logger.info("Loading M1 checkpoint")
        m1_ckpt_path = hdfs_get_cache(args.m1_ckpt_path)
        m1_model = m1_model.load_from_checkpoint(m1_ckpt_path)

    trainer = cfg.trainer
    trainer.strategy.connect(m1_model)
    trainer.strategy.setup(trainer)

    # prediction dataset
    batch_size = 8
    prompt_path = data_bucket(
        "data/prompts/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv",
        cache=True,
    )
    predict_dataset = inference_dataset_from_prompt(
        prompt_path=prompt_path,
        conditions="style_text,lyrics_tokens",
        max_items=None,
        batch_size=batch_size,
        run_combinations=False,
        lyrics_max_seq_len=400,
        enable_punctuation=True,
    )
    pl_datamodule = LyricsDataModule(predict_dataset, num_workers=0)
    dataloader = pl_datamodule.train_dataloader()
    batch = next(iter(dataloader))

    inference = Inference(m1_model)

    inference = inference.to("cuda")

    batch = {k: v.to("cuda") if type(v) == torch.Tensor else v for k, v in batch.items()}

    audio = inference.generate(batch)

    for batch_idx in range(batch_size):
        torchaudio.save(f"test_{batch_idx}.wav", audio[batch_idx][None, :].cpu(), inference.sample_rate)
