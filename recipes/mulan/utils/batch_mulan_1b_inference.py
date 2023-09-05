import argparse
import random
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from recipes.mulan.modules.pl_module import LitMuLanModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

mulan_ckpt1 = "/mnt/bn/dguo-mir-workspace/Models/mulan_1b_multi_dset/mulan-step=123200-median_rank_0=114-kaggle.ckpt"
mulan_ckpt2 = "/mnt/bn/dguo-mir-workspace/Models/mulan_1b_multi_dset/mulan-step=130400-median_rank_2=64-kaggle.ckpt"
mulan_ckpt3 = "/mnt/bn/dguo-mir-workspace/Models/mulan_1b_multi_dset/mulan-step=049600-median_rank_1=203-kaggle.ckpt"
mulan_ckpt4 = "/mnt/bn/dguo-mir-workspace/Models/mulan_1b_g4/mulan-step=044800-median_rank_1=170-kaggle.ckpt"
mulan_ckpt5 = "/mnt/bn/dguo-mir-workspace/Models/mulan_1b_g4/mulan-step=020800-median_rank_1=183-kaggle.ckpt"
mulan_ckpt6 = "/mnt/bn/dguo-mir-workspace/Models/mulan_1b_gpt_all/mulan-step=036000-median_rank_0=127-kaggle.ckpt"

mulan_ckpt7 = "/mnt/bn/mm-data/user/mulan_exp/MuLan_large/mulan_0724_mixall_25hz/checkpoints/mulan-step=014000-median_rank_1=160-kaggle.ckpt"
N_DEVICES = 8

target_emb_path_1 = "/opt/tiger/arnold_starter/aigc/embeds_mulan1b_114"
os.makedirs(target_emb_path_1, exist_ok=True)

target_emb_path_2 = "/opt/tiger/arnold_starter/aigc/embeds_mulan1b_64"
os.makedirs(target_emb_path_2, exist_ok=True)

target_emb_path_3 = "/opt/tiger/arnold_starter/aigc/embeds_mulan1b_203"
os.makedirs(target_emb_path_3, exist_ok=True)

target_emb_path_4 = "/opt/tiger/arnold_starter/aigc/embeds_mulan1b_g4_170"
os.makedirs(target_emb_path_4, exist_ok=True)

target_emb_path_5 = "/opt/tiger/arnold_starter/aigc/embeds_mulan1b_g5_183"
os.makedirs(target_emb_path_5, exist_ok=True)

target_emb_path_6 = "/opt/tiger/arnold_starter/aigc/embeds_mulan1b_gpt_all_127"
os.makedirs(target_emb_path_6, exist_ok=True)

target_emb_path_7 = "/opt/tiger/arnold_starter/aigc/embeds_mulan1b_mix_170"
os.makedirs(target_emb_path_7, exist_ok=True)

def collect_files():
    aed_folder = "/mnt/bd/sami-mm-music-shard1/Dataset/aed_700K"
    aed_files = os.listdir(aed_folder)
    aed_files = [f"{aed_folder}/{f}" for f in aed_files]

    dataset = []
    with open(
        "/mnt/bd/litang-lq-music-npy11/15m_filter_trim.txt",
        "r",
    ) as fp:
        for line in fp:
            duration = int(line.split("|")[1]) // 24000
            if duration < 30 or duration > 600:
                continue
            dataset.append(line.strip())

    n = len(dataset) // 3
    for i in range(n):
        TARGET_DIR = "/mnt/bd/sami-mm-music-shard1/Dataset/mcc"
        audio_path, audio_length, audio_sr = dataset[i].strip().split("|")
        audio_name = audio_path.split("/")[-1]
        audio_id = audio_name.split(".")[0]
        audio_suffix = audio_name.replace(audio_id, "")
        target_file = f"{TARGET_DIR}/{audio_name}_clips{audio_suffix}"
        dataset[i] = target_file
    for i in range(n, len(dataset)):
        TARGET_DIR = "/mnt/bd/sami-mm-music-shard2/Dataset/mcc"
        audio_path, audio_length, audio_sr = dataset[i].strip().split("|")
        audio_name = audio_path.split("/")[-1]
        audio_id = audio_name.split(".")[0]
        audio_suffix = audio_name.replace(audio_id, "")
        target_file = f"{TARGET_DIR}/{audio_name}_clips{audio_suffix}"
        dataset[i] = target_file
    return sorted(aed_files) + sorted(dataset)




class CustomerDataset_extracted_feature(Dataset):
    def __init__(self, audio_paths):
        self.audio_paths = audio_paths

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        try:
            path = self.audio_paths[idx]
            wavs = np.load(path)
            if wavs.ndim == 1:
                wavs = wavs.reshape([-1, 240000])
            if wavs.dtype == "int16":
                wavs = (wavs / 32768.).astype(np.float32)
            assert np.max(wavs) <= 1 and np.min(wavs) >= -1
            while wavs.shape[0] < 3:
                wavs = np.concatenate([wavs, wavs[:1, :].copy()], axis=0)
            return wavs, path
        except:
            idx = random.randint(0, len(self.audio_paths) - 1)
            return self.__getitem__(idx)


def run_inference(audio_dset, ckpt_path, target_emb_path, device_id):

    device = "cuda:{}".format(device_id)
    eval_dset = CustomerDataset_extracted_feature(audio_dset)
    eval_loader = DataLoader(
        eval_dset,
        num_workers=20,
        shuffle=False,
        batch_size=60,
        pin_memory=True,
        drop_last=False,
    )

    # ckpt_path = "/mnt/bn/dguo-mir-workspace/Models/mulan_1b/mulan-step=006400-median_rank_1=115-kaggle.ckpt"
    mulan_model = LitMuLanModule.load_from_checkpoint(ckpt_path)

    print("Loaded MuTWrapper: the parameter values are:")
    cnt = 0
    for k, v in mulan_model.music_encoder.mut.named_parameters():
        cnt += 1
        if cnt == 32:
            break 
        print(np.around(torch.mean(v).item(), 8), k, v.shape)
    print("Sanity Check of MuTWrapper done")

    # audio tower
    mulan_model.music_encoder.eval()
    mulan_model.music_encoder.to(device)
    mulan_model.music_encoder.mut.manually_to_device(device)
    # text tower
    mulan_model.text_encoder.eval()
    mulan_model.text_encoder.to(device)

    pytorch_total_params = sum(p.numel() for p in mulan_model.parameters())
    print("Generator Params: {:.4f}M".format(pytorch_total_params / 1e6))

    with torch.no_grad():
        for i, (loaded_data, current_path) in tqdm(enumerate(eval_loader)):
            filename = os.path.basename(current_path[0])
            target_path = os.path.join(target_emb_path, filename)
            if os.path.exists(target_path):
                continue
            [b, n, t] = loaded_data.shape
            wavs = loaded_data.reshape([b * n, 1, t]).float().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    # with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable :
                    mulan_embeds = mulan_model.music_encoder(
                        wavs.to(device), spec_aug=False
                    )
            mulan_embeds = mulan_embeds.detach().cpu().numpy()
            np.save(target_path, mulan_embeds)  # [t, n_codebook]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=24)
    args = parser.parse_args()

    shard_id = args.shard_id
    n_shards = args.n_shards
    dataset = collect_files()

    dataset = dataset[shard_id::n_shards]
    if shard_id % 2 == 0:
        dataset = dataset[::-1]

    audio_dataset = []
    for row in dataset:
        if os.path.exists(row):
            audio_dataset.append(row)
    print(f"processing {len(audio_dataset)} audios")

    run_inference(audio_dataset, mulan_ckpt6, target_emb_path_6, shard_id % N_DEVICES)
