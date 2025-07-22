import torch, os
from recipes.umm2.loaders.base import BaseModelLoader, local_zero_first
import sys, importlib
from recipes.umm2.benchmark.dataset import EvalDataset
import numpy as np
import math

class ModelLoader(BaseModelLoader):
    def __init__(
        self,
        ckpt_path,
        cache_dir,
        device,
    ):
        super().__init__(ckpt_path, cache_dir)
        self.device = device

    def load_model(self, pl_module_cls):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        if not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)
        with local_zero_first():
            local_path = self.ensure_hdfs_ckpt_is_local(self.ckpt_path, self.cache_dir)
            model = pl_module_cls.load_from_checkpoint(local_path, map_location="cpu").to(self.device).eval()
            
        return {"pl_module": model}
            

def init_model(ckpt_path, cache_dir, device, pl_module_string):
    loader = ModelLoader(ckpt_path, cache_dir=cache_dir, device=device)
    module_name, cls_name = pl_module_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    pl_module_cls = getattr(module, cls_name)
    model = loader.load_model(pl_module_cls)
    return model

@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def example_wav2token(audio, pl_module, slice_method='full', chunk_size=60, sample_rate=24000):
    if slice_method == 'full':
        result_dict = pl_module.wav2token(audio)
        vq_latent = result_dict["latent"]   # [B, T, N]
        vq_ids = result_dict["vq_ids"].squeeze(-1)      # [B, T, R] / [B, T]
    
    else:
        vq_ids, vq_latent = [], []
        n_samples = audio.shape[-1]
        target_audio_length = float(n_samples) / sample_rate
        if slice_method == 'even':
            chunk_num = math.ceil(target_audio_length / chunk_size)
            chunk_size = math.ceil(target_audio_length / chunk_num)
        elif slice_method == 'max':
            chunk_size = chunk_size
        
        st = 0
        while st < n_samples:
            st_sample, et_sample = int(st*sample_rate), int((st+chunk_size)*sample_rate)
            # merge the tail if the remaining chunk is too short (<5s)
            if n_samples - et_sample < sample_rate * 5 or audio[..., et_sample:].shape[-1] < sample_rate * 5:
                et_sample = n_samples
            chunk_result_dict = pl_module.wav2token(audio[..., st_sample:et_sample])
            vq_ids.append(chunk_result_dict["vq_ids"].squeeze(-1))
            vq_latent.append(chunk_result_dict["latent"])
            if et_sample >= n_samples:
                break
            st += chunk_size
        vq_ids = torch.cat(vq_ids, dim=-1)
        vq_latent = torch.cat(vq_latent, dim=-2)

    
    print(f"{vq_latent.shape=}")    # [B, T, H]
    print(f"{vq_ids.shape=}")       # [B, T]
    print(vq_ids[0, :100])
    return vq_ids



if __name__ == "__main__":

    # 25Hz, 15bit x 2 (RVQ)
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/RVQ/RVQ_RP_2x32768_lr3e-5_30k_pitchpdt_re2/checkpoints/step=0250000.ckpt"

    # 50Hz, 15bit x 1 (VQ)
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/VQ/RP_50Hz_1x32768_lr1e-4_100k_bs4.5/checkpoints/step=0220000.ckpt"

    # 50Hz, new ckpt
    ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_FT25Hz_lr1e-4_cyc100k_bs4.5_VQ-RP_CTC-25Hz/checkpoints/step=0220000.ckpt"


    cache_dir = "./module_cache/umm2/"
    local_rank = 0
    device = torch.device(f"cuda:{local_rank}")

    model = init_model(ckpt_path, cache_dir, device, "recipes.umm2.modules.stages.stage3.Stage3RVQ")
    pl_module = model["pl_module"]
    
    import torchaudio
    audio, sr = torchaudio.load("inp071.generated.wav")
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000)
    # take the first channel
    audio = audio[0].unsqueeze(0).cuda()    
    print(audio.shape, audio.shape[-1] / 24000)

    # full length extraction
    print("="*20, "full length extraction", "="*20)
    full_vq_ids = example_wav2token(audio, pl_module)

    print("="*20, "chunk-wise extraction", "="*20)
    chunk_vq_ids = example_wav2token(audio, pl_module, slice_method='even', chunk_size=45)

    print("locality (full v.s. chunk-wise)")
    print(torch.sum(full_vq_ids == chunk_vq_ids) / torch.prod(torch.tensor(full_vq_ids.shape)))

    with open("test1_full.txt", "w") as f:
        f.writelines("\n".join(full_vq_ids[0].cpu().numpy().astype(str).tolist()))

    with open("test1_chunk.txt", "w") as f:
        f.writelines("\n".join(chunk_vq_ids[0].cpu().numpy().astype(str).tolist()))
    # config = pl_module.model.stages[0].config
    # text_tokenizer = pl_module.hparams.extra_params["tokenizer"]
    # num_workers = 0
    # batch_size = 1
    # l2 = EvalDataset(
    #     data_id=7602,
    #     url_pattern=None,
    #     frame_rate=config.frame_rate,
    #     sample_rate=config.sample_rate,
    #     num_worker=num_workers,
    #     batch_size=batch_size,
    #     min_duration=60,
    #     max_duration=600,
    #     tokenizer=text_tokenizer,
    # )

    # cnt = 0
    # for data in l2.dataloader:
    #     cnt += 1
    #     print(l2.data_id, cnt, data.keys(), data["audio"].shape, data["token"].shape)
    #     result_dict = pl_module.wav2token(data["audio"].cuda())
    #     print(result_dict.keys())
    #     print(result_dict["latent"].shape, result_dict["vq_ids"].shape)

    #     if cnt >= 5:
    #         break
