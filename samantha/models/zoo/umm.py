from typing import Dict

import torch

from recipes.umm.models.umm_mkii import UMMResult
from recipes.umm.requires.model_initializer import init_stage3
from samantha.models.base import LightningModuleBase
from samantha.utils.hdfs_tools import ARNOLD_REGION


class UMM(LightningModuleBase):
    sample_rate: int = 24000
    frame_rate: int = 25
    output_dim: int = 1024
    precision = torch.float32

    model_paths: Dict[str, str] = {
        "en-stage1": {
            "US": "hdfs://harunava/home/byte_speech_sv/zongyu.yin/logs/umm/umm_stage1_full_dw1-1-1_rq4096x16x8/checkpoints/step=100000.ckpt",  # noqa
            "CN": "hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/umm_stage1_full_dw1-1-1_rq4096x16x8/checkpoints/step=100000.ckpt",  # noqa
        },
        "en-stage2": {
            "US": "hdfs://harunava/home/byte_speech_sv/zongyu.yin/logs/umm/stage2_music/checkpoints/step=0030000-loss=0.552.ckpt",  # noqa
            "CN": "hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/stage2_music/checkpoints/step=0030000-loss=0.552.ckpt",  # noqa
        },
        "en-stage3": {
            "US": "hdfs://harunava/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt",  # noqa
            "CN": "hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt",  # noqa
        },
        "zh-stage3": {
            "CN": "hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/umm_stage3_zh_dw1-1-0_wordpiece_vq32768x16-layer12/checkpoints/step=070000.ckpt"  # noqa
        },
        "en-zh-stage3": {
            "CN": "hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt"  # noqa
        },
    }

    def __init__(
        self,
        model_version: str,
        layer_idx: int = 12,
        finetune: bool = False,
        cache_dir: str = ".cache",
    ):
        super().__init__()
        self.model_version = model_version
        self.layer_idx = layer_idx
        self.finetune = finetune

        self.model = self.get_model(model_version, cache_dir)

        if not finetune:
            self.freeze()
            self.eval()

    def get_model(self, model_version: str, cache_dir: str):
        assert model_version in self.model_paths
        assert ARNOLD_REGION in self.model_paths[model_version]

        model = init_stage3(
            self.model_paths[model_version][ARNOLD_REGION],
            local_rank=0,
            cache_dir=cache_dir,
        )
        return model["Stage3"].model

    @torch.cuda.amp.autocast(enabled=False, dtype=precision)
    def forward(self, audio: torch.Tensor) -> UMMResult:
        if not self.finetune:
            self.model.eval()

        with torch.set_grad_enabled(self.finetune):
            return self.model.wav2hidden_states(audio, self.layer_idx)


class UMMStage3_Layer12PostVQ(LightningModuleBase):
    sample_rate: int = 24000
    frame_rate: int = 25
    output_dim: int = 32  # 1024
    layer_idx: int = 12
    precision = torch.float32
    model_path = "hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt"  # noqa

    def __init__(self, finetune: bool = False, cache_dir: str = ".cache"):
        super().__init__()
        self.finetune = finetune

        self.model = init_stage3(
            self.model_path[ARNOLD_REGION], local_rank=0, cache_dir=cache_dir
        )
        self.model = self.model["Stage3"].model

        if not finetune:
            self.freeze()
            self.model.eval()

    def forward_layers(
        self, hidden_states: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        # TODO: interface change in next version :^)

        hidden_states = self.model.shared_encoder.dropout(hidden_states)
        position_embeddings = self.model.shared_encoder.embed_positions(hidden_states)

        for i, layer in enumerate(self.model.shared_encoder.layers):
            if i == layer_idx:
                vq_states, vq_ids, vq_loss, vq_emb = self.model.vq(hidden_states)
                return vq_emb

            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        return hidden_states

    @torch.cuda.amp.autocast(enabled=False, dtype=precision)
    def forward(self, x: torch.Tensor) -> UMMResult:
        if not self.finetune:
            self.model.eval()

        with torch.set_grad_enabled(self.finetune):
            embed = self.model.wav2embed(x)
            out = self.forward_layers(embed, self.layer_idx)
            return UMMResult(out)
