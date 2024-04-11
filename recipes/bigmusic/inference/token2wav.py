import os

import torch
from recipes.diffusion.models.diffusion_model.utils import run_diffusion
from recipes.diffusion.models.diffusion_model.utils import init_diffusion
from recipes.diffusion.inference_from_text_lyrics2song import init_sampler
from recipes.diffusion.models.vocoder_model.utils import init_vocoder


class Token2Wav:
    diffusion_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wtl/diffusion/model_14_30s_finetune/checkpoints/last-minimal.ckpt"
    vocoder_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wtl/vocoder/soundstream-step=374999-val_sdr=12.9557.ckpt"
    cache_root_dir = os.path.join(
        os.environ["DUMP_DIR"],
        "ai_music/20240312.symbolic.dump/"
    )

    def __init__(self):
        diffusion_cache_dir = os.path.join(self.cache_root_dir, "diffusion")
        vocoder_cache_dir = os.path.join(self.cache_root_dir, "vocoder")
        self.requires = {
            **init_diffusion(self.diffusion_path, 0, diffusion_cache_dir),
            **init_sampler(None, 0, None),
            **init_vocoder(self.vocoder_path, 0, vocoder_cache_dir),
        }

    def run(self, tokens):
        return run_diffusion(
            self.requires,
            torch.tensor(tokens, device="cuda:0")[None, ...],
            {}
        )[0].cpu().numpy()