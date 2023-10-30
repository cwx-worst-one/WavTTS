import torch
import torchaudio
from torch.utils.data import DataLoader

assert torch.cuda.is_available()

from tqdm import tqdm
import os

from recipes.datasets.mcc.mix import MixWebDataModule, collate_audio
from recipes.umm.modules.lit_module import Stage1, Stage2, Stage3
import numpy as np
import librosa
from torchaudio_augmentations import Compose
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)
import fairseq
import torch.nn.functional as F




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
        audio = self.base_transform(audio)
        # tokens = self.umm_model.wav2token(audio).cpu().to(torch.int).numpy()
        # 保存为npy格式
        target_path = os.path.join(self.target_dir, os.path.relpath(self.file_list[index], start=self.root_dir)) + '.npy'
        # np.save(target_path, )
        return audio, target_path



@torch.no_grad()
def evaluate(dataloader, model, cfg, task):

    # prepare dir
    for subdir, dirs, files in os.walk(dataloader.dataset.root_dir):
        for file in files:
            dest_subdir = subdir.replace(dataloader.dataset.root_dir, dataloader.dataset.target_dir)
            os.makedirs(dest_subdir, exist_ok=True)

    for i, (audio, target_path) in tqdm(enumerate(dataloader)):
        audio = audio.to("cuda")
        if audio.dim() == 3:
            audio = audio.squeeze(dim=1)
        if task.cfg.normalize:
            audio = torch.stack([F.layer_norm(wav, wav.shape) for wav in audio])
        with torch.no_grad():
            results = model.extract_features(audio, padding_mask=None, output_layer=9999, ret_layer_results=True)
            if isinstance(results, tuple):
                f = results[0]
            layer_results = f[1]
            layer_results = [x.transpose(0, 1) for x, _ in layer_results]
            bn = torch.cat(layer_results, 0).cpu().numpy()
        # bn = wav2vqbn(umm_model, audio).cpu().numpy()  # modify
        # print(tokens, '   ', target_path)
        np.save(target_path[0], bn)
        


if __name__ == "__main__":

    ckpt_path = "/mnt/bn/cyz-lq-nas/model_cache/wavlm/chunk_one_million_large_online_model_largebs_no1p/checkpoint_last.pt"  
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = model[0].to("cuda").eval()

    dataset = AudioDataset(root_dir='/mnt/bn/cyz-lq-nas/s3prl_datasets/LibriSpeech', 
                    target_dir='/mnt/bn/cyz-lq-nas/s3prl_gendir/LibriSpeech/fix/WavLM_allbn') # modify
                    # 32维 for vqbn
    dataloader = DataLoader(dataset, num_workers=24)


    print(ckpt_path)
    eval_out_libritts = evaluate(dataloader, model, cfg, task)
