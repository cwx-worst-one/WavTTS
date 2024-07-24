from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from pytorch_lightning.utilities import grad_norm
from recipes.research.audio_codec.zoo import AudioCodec_7c355ea_64l

from recipes.research.diff.diff import DiffConfig
from recipes.research.diff.prod import load_diffusion_model
from samantha.data.audio.types import AudioDataResult, AudioMeta, SegmentInfo
from samantha.models.base import DefaultTrainingBaseModule
from samantha.optim.lr_scheduler.tri_stage_lr import TriStageLR
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)



class DiffLora(DefaultTrainingBaseModule):


    def __init__(self, config: DiffConfig):
        super().__init__()
        self.config = config

        self.model = load_diffusion_model(
            "5c5f7aa",
            ckpt_path="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/logs/diff/default/5c5f7aa/2024-07-09/05-08-40/checkpoints/step=530000.ckpt",
            device="cpu",
        )
        self.model.load_audio_codec(AudioCodec_7c355ea_64l(), device="cpu")
        self.x_latent_dim = self.model.x_latent_dim

        self.model = self.model.eval()
        self.model.freeze()

        lora_config = LoraConfig(
            r=8,
            target_modules=["Wqkv", "out_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)

    def setup(self, stage: Optional[str] = None, device: Optional[torch.device] = None):
        pass

    def on_before_optimizer_step(self, optimizer):
        if self.trainer.global_step % 500 == 0:
            norms = grad_norm(self, norm_type=2)
            self.log_dict(norms)

    def step(self, batch: AudioDataResult, batch_idx: int, return_loss: bool):
        result = self.model.step(batch, batch_idx, return_loss)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=1e-8,
            weight_decay=self.config.weight_decay,
        )

        scheduler = TriStageLR(
            optimizer,
            self.config.learning_rate,
            self.config.min_lr,
            self.config.warmup_steps,
            self.config.hold_steps,
            self.config.decay_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

if __name__ == "__main__":
    config = DiffConfig(
        sample_rate=44100, max_duration=30, n_layer=4, n_embd=1024, n_head=16
    )
    model = DiffLora(config).to("cuda")
    model.setup()
    print(model.summarize())

    batch_size = 2

    x = torch.randn(
        batch_size,
        model.model.audio_codec.n_channels,
        config.max_duration * config.sample_rate,
        device=model.device,
    )

    segment_info = SegmentInfo(
        meta=AudioMeta(path=None, duration=30, sample_rate=config.sample_rate),
        data_type="music_vocal",
        seek_time=0.0,
        n_frames=x.shape[2],
        total_frames=x.shape[2],
        sample_rate=config.sample_rate,
        channels=2,
        lyrics=None,
    )
    index = {
        "keywords": "upbeat, technology, technological, pulsing, lively, kawaii, japan, cute, bubblegum, bouncy",
        "genres": "country",
        "instruments": "drums, guitar, piano",
        "bpm": "121",
        "description": "Bouncy and cute",
        "text": "upbeat, technology, technological",
    }

    batch = AudioDataResult(
        audio=x,
        shard=None,
        key=None,
        segment_info=[segment_info] * batch_size,
        index=[index] * batch_size,
    )

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        result = model.step(batch, batch_idx=0, return_loss=True)
