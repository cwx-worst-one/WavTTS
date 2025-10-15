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
            model = pl_module_cls.load_from_checkpoint(local_path, strict=False, map_location="cpu").to(self.device).eval()

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


def init_model(ckpt_path, pl_module_string , cache_dir="./modules/umm2/", device=None, requires=None):
    loader = ModelLoader(ckpt_path, cache_dir=cache_dir, device=device, requires=requires)
    module_name, cls_name = pl_module_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    pl_module_cls = getattr(module, cls_name)
    model = loader.load_model(pl_module_cls)
    return model["pl_module"]


def init_umm2(hpath, local_rank, pl_module_string="recipes.umm2.modules.stages.stage2.Stage2", cache_dir=None):

    model = init_model(
        ckpt_path = hpath,
        pl_module_string = pl_module_string,
        device = torch.device(f"cuda:{local_rank}")
    )
    return {"UMM2_stage2": model}


def load_example_audio(audio_path=None):
    if audio_path is None:
        # audio_path = "/mnt/bn/music-llm-nas-lq/qinxin/bak/inp071.generated.wav"
        # if not os.path.exists(audio_path):
        #     audio_path = "/mnt/hdfs/qinxin.025/testset/token2wav/inp071.generated.wav"
        # if not os.path.exists(audio_path):
        os.system("hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/testset/token2wav/inp071.generated.wav .")
        audio_path = "inp071.generated.wav"

    audio, sr = torchaudio.load(audio_path)
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000)

    audio = audio[0].unsqueeze(0).unsqueeze(1).cuda()   # [B, 1, T]
    print("audio duration", audio.shape[-1] / 24000)

    return audio

if __name__ == "__main__":
    # RVQ4 (after umm2 refactor)
    # hpath = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/2025072916_UMM-stage3_64GPU_fused_attnmask_model_RVQRP3_14bit/checkpoints/step=0080000.ckpt"
    # RVQ4 (before umm2 refactor)
    hpath = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/RVQ/RVQ_RP_4x16384_lr3e-5_30k_bs5_a100/checkpoints/step=0200000.ckpt"
    local_rank = 0
    model_dict = init_umm2(hpath=hpath, local_rank=local_rank,)
    requires = ["token"] #, "loss"]
    model = model_dict["UMM2_stage2"]


    audio = load_example_audio()
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
    print(full_output_dict["token"])

    chunk_tokens = chunk_output_dict["token"]
    with open("test1_chunk.txt", "w") as f:
        f.writelines("\n".join(chunk_tokens[0].cpu().reshape(-1).numpy().astype(str).tolist()))

    full_tokens = full_output_dict["token"]
    with open("test1_full.txt", "w") as f:
        f.writelines("\n".join(full_tokens[0].cpu().reshape(-1).numpy().astype(str).tolist()))

