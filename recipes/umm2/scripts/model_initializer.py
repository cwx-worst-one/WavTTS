import torch, os
from recipes.umm2.loaders.base import BaseModelLoader, local_zero_first
import sys, importlib
from recipes.umm2.modules.utils import get_task_losses
import math
import contextlib
import os
import torchaudio
import torch


class ModelLoader(BaseModelLoader):
    def __init__(
        self,
        ckpt_path,
        requires,
        cache_dir,
        device,
    ):
        super().__init__(ckpt_path, cache_dir)
        self.device = device

        # currently support {token, tag, pre-vq/post-vq latent, loss_dict, ...}
        self.requires = requires

    def load_model(self, pl_module_cls):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        if not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)

        with local_zero_first():
            local_path = self.ensure_hdfs_ckpt_is_local(self.ckpt_path, self.cache_dir)
            model = pl_module_cls.load_from_checkpoint(local_path, strict=False).to(self.device).eval()

        self.pl_module = model
        return {"pl_module": model}
    
    def load_encoder_only(self, model_cls):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        return NotImplementedError
    

def init_stage3(hpath, local_rank, requires=[], cache_dir=None):
    """Init function for standard Stage3 UMM backbone."""
    from recipes.umm2.modules.stages.stage3 import Stage3RVQ

    device = torch.device(f"cuda:{local_rank}")
    loader = ModelLoader(hpath, requires=requires, cache_dir=cache_dir, device=device)
    # module_name, cls_name = pl_module_string.rsplit(".", 1)
    # module = importlib.import_module(module_name)
    # pl_module_cls = getattr(module, cls_name)
    model = loader.load_model(Stage3RVQ)
    return model["pl_module"]

def load_example_audio(audio_path=None):
    if audio_path is None:
        audio_path = "/mnt/bn/music-llm-nas-lq/qinxin/bak/inp071.generated.wav"
        audio_path = "/mnt/hdfs/qinxin.025/testset/token2wav/inp071.generated.wav"

    audio, sr = torchaudio.load(audio_path)
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000)

    audio = audio[0].unsqueeze(0).unsqueeze(1).cuda()   # [B, 1, T]
    print("audio duration", audio.shape[-1] / 24000)

    return audio

if __name__ == "__main__":
    # audio shape: [B, 1, T] (24kHz)
    audio = load_example_audio()

    # 25Hz RVQ4
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/RVQ/RVQ_RP_4x16384_lr3e-5_30k_bs5_a100/checkpoints/step=0200000.ckpt"

    # 50Hz finetune stage3
    ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_FT25Hz_lr1e-4_cyc100k_bs4.5_VQ-RP_CTC-25Hz/checkpoints/step=0220000.ckpt"
    # ckpt_path = "hdfs://haruna/home/byte_speech_sv/ju-chiang.wang/umm_tag/karaoke/umm_stage4_artist_weight_mel_chroma_ctc_vq/checkpoints/step=0320000.ckpt"
    
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_FT25Hz_lr1e-4_cyc100k_bs4.5_VQ-RP_CTC-25Hz_base2w_reweight/checkpoints/step=0220000.ckpt"

    # # 50Hz stage2
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2/50Hz_lr2e-4_cyc200k_bs5_fixall/checkpoints/step=0255000.ckpt"
    cache_dir = "./module_cache/umm2/"
    local_rank = 0

    # requires = ["loss", "token", "latent"] # stage3 / stage4
    # requires = ["loss", "token", "latent", "tag"] #  stage4
    # requires = ["latent"]   # stage2
    requires = ["token"]
    model = init_stage3(ckpt_path, local_rank, requires=requires, cache_dir=cache_dir)

    
    # chunk-wise inference (recommended)
    print("======= chunk-wise inference =======")
    chunk_output_dict = model.wav2requires(audio, requires=requires, slice_method="max", chunk_size=45)
    for k in chunk_output_dict.keys():
        print(k)
        print(chunk_output_dict[k].shape if isinstance(chunk_output_dict[k], torch.Tensor) and chunk_output_dict[k].ndim > 1 else chunk_output_dict[k])
    print(chunk_output_dict["token"])

    # full-length inference (not recommended)
    print("======= full-length inference =======")
    full_output_dict = model.wav2requires(audio, requires=requires, slice_method="full", chunk_size=None)
    for k in full_output_dict.keys():
        print(k)
        print(full_output_dict[k].shape if isinstance(full_output_dict[k], torch.Tensor) and full_output_dict[k].ndim > 1 else full_output_dict[k])


    chunk_tokens = chunk_output_dict["token"]
    full_tokens = full_output_dict["token"]

    with open("test_full.txt", "w") as f:
        f.writelines("\n".join(full_tokens[0].cpu().reshape(-1).numpy().astype(str).tolist()))
    with open("test_chunk.txt", "w") as f:
        f.writelines("\n".join(chunk_tokens[0].cpu().reshape(-1).numpy().astype(str).tolist()))
