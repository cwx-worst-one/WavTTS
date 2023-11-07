import torch
import torchaudio
from torch.utils.data import DataLoader

assert torch.cuda.is_available()

from tqdm import tqdm
import os

from recipes.umm.modules.lit_module_mk3 import USMStage2
import numpy as np
import librosa
from torchaudio_augmentations import Compose
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)




class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, target_dir):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.file_list = self.get_audio_files(self.root_dir)
        self.base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        self.base_transform = Compose(self.base_transforms)

    def get_audio_files(self, directory):
        # 递归获取所有音频文件
        file_list = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(('.wav', '.flac')):
                    file_list.append(os.path.join(root, filename))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio, sr = librosa.load(self.file_list[index], mono=True, sr=None)
        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)
        # tokens = self.umm_model.wav2token(audio).cpu().to(torch.int).numpy()
        # 保存为npy格式
        target_path = os.path.join(self.target_dir, os.path.relpath(self.file_list[index], start=self.root_dir)) + '.npy'
        # np.save(target_path, )
        return audio, target_path



@torch.no_grad()
def evaluate(dataloader, model):

    # prepare dir
    for subdir, dirs, files in os.walk(dataloader.dataset.root_dir):
        for file in files:
            dest_subdir = subdir.replace(dataloader.dataset.root_dir, dataloader.dataset.target_dir)
            os.makedirs(dest_subdir, exist_ok=True)

    for i, (audio, target_path) in tqdm(enumerate(dataloader)):
        audio = audio.to("cuda")
        if audio.shape[1] > 16000*60:
            continue
        features = model.extract_features(audio, dtype=torch.bfloat16)
        features = features[27::] #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        features = torch.cat(features, 0).cpu().numpy() 
        # features = features[31].cpu().numpy() 
        # print(tokens, '   ', target_path)
        np.save(target_path[0], features)
        


if __name__ == "__main__":

    ckpt_path = '/mnt/bn/cyz-lq-nas/model_cache/usm/pl_asr_2B_ft.ckpt'
    # ckpt_path = '/mnt/bn/cyz-lq-nas/model_cache/usm/usm_stage2.ckpt'
    model = USMStage2.load_from_checkpoint(ckpt_path).eval().cuda()

    # dataset_list = ['LibriSpeech', 'Vox1', 'IEMOCAP']
    dataset_name = 'IEMOCAP'

    root_dir = os.path.join('/mnt/bn/cyz-lq-nas/s3prl_datasets', dataset_name)
    target_dir = os.path.join('/mnt/bn/cyz-lq-nas/s3prl_gendir/USM_pl_asr_2B_ft_b27e31i1', dataset_name)

    # root_dir = os.path.join('/mnt/bn/cyz-lq-nas/s3prl_datasets', dataset_name)
    # target_dir = os.path.join('/mnt/bn/cyz-lq-nas/s3prl_gendir/USM_stage2_b27e31i1', dataset_name)

    dataset = AudioDataset(root_dir=root_dir, target_dir=target_dir) # modify
    dataloader = DataLoader(dataset, num_workers=12)

    print(ckpt_path)
    print(dataset_name)
    evaluate(dataloader, model)
