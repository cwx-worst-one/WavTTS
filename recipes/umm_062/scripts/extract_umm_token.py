import torch
import torchaudio
from torch.utils.data import DataLoader

assert torch.cuda.is_available()

from tqdm import tqdm
import os

from recipes.datasets.mcc.mix import MixWebDataModule, collate_audio
from recipes.umm_062.modules.lit_module import Stage3
import numpy as np
import librosa
from torchaudio_augmentations import Compose
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)

@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def wav2token_ws(model, wav):
    if wav.dim() == 3:
        wav = wav.squeeze(dim=1)
    wav = model.pad_audio(wav.float())
    feature = model.preprocessing(wav)["mel"]
    audio_feature = model.audio_encoder(feature)
    hidden_states = model.encoder_input_dropout(audio_feature)
    position_embeddings = model.embed_positions(hidden_states)
    hidden_states_list = []
    for i, layer in enumerate(model.encoder_layers):
        hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        if i <= model.config.vq_layer_idx-1:
            hidden_states_list.append(hidden_states)

        if i == model.config.vq_layer_idx-1:
            # weighted sum
            if model.use_weighted_sum:
                hidden_states = model._weighted_sum(hidden_states_list)
            hidden_states = model.vq_proj_in(hidden_states)
            vq_embs, vq_ids, vq_loss = model.vq(hidden_states)
            return vq_ids
    return vq_ids


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
        audio, sr = librosa.load(
            self.file_list[index], mono=True, sr=None)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        audio = self.base_transform(audio)
        # tokens = self.umm_model.wav2token(audio).cpu().to(torch.int).numpy()
        # 保存为npy格式
        target_path = os.path.join(self.target_dir, os.path.relpath(self.file_list[index], start=self.root_dir)) + '.npy'
        # np.save(target_path, )
        return audio, target_path



@torch.no_grad()
def evaluate(dataloader, umm_model):
    # for i, batch in tqdm(enumerate([1,2,3,4])):
    #     # wav = batch["audio"].to("cuda")
    #     wav = torch.rand([1, 10000]).to("cuda")
    #     if wav.dim() == 3:
    #         wav = wav.squeeze(dim=1)
    #     tokens = umm_model.wav2token(wav)
    #     print(tokens)

    # prepare dir
    for subdir, dirs, files in os.walk(dataloader.dataset.root_dir):
        for file in files:
            dest_subdir = subdir.replace(dataloader.dataset.root_dir, dataloader.dataset.target_dir)
            os.makedirs(dest_subdir, exist_ok=True)

    for i, (audio, target_path) in tqdm(enumerate(dataloader)):
        audio = audio.to("cuda")
        # tokens = umm_model.wav2token(audio).cpu().to(torch.int).numpy()
        tokens = wav2token_ws(umm_model, audio).cpu().to(torch.int).numpy()
        # print(tokens, '   ', target_path)
        np.save(target_path[0], tokens)
        # print(f'Processed file {i+1}, saved at {target_path}')
        # print(audio.shape)
        
    # for subdir, dirs, files in tqdm(os.walk(source_dir)):
    #     for file in files:
    #         # 只处理.wav和.flac文件 
    #         if file.endswith(".wav") or file.endswith(".flac"):
    #             # 获取文件的完整路径
    #             file_path = os.path.join(subdir, file)
                
    #             # 使用pydub读取音频（确保存在ffmpeg）
    #             audio, sr = librosa.load(file_path, mono=True)

    #             # if sr != 24000:
    #             #     audio = librosa.resample(audio, sr, 24000)

    #             audio = base_transform(audio).to("cuda")

    #             tokens = umm_model.wav2token(audio).cpu().to(torch.int).numpy()

    #             # 创建对应的目标文件路径
    #             dest_subdir = subdir.replace(source_dir, dest_dir)
    #             os.makedirs(dest_subdir, exist_ok=True)
    #             dest_file_path = os.path.join(dest_subdir, file.rsplit(".", 1)[0] + ".npy")


    #             # 保存.npy
    #             # print(tokens, tokens.shape)
    #             # print(dest_file_path)
    #             np.save(dest_file_path, tokens)


if __name__ == "__main__":

    ckpt_path = "/mnt/bn/cyz-lq-nas/model_cache/umm/yuanzhe_v0.3.2_75k.ckpt"
    umm_model = Stage3.load_from_checkpoint(ckpt_path).model.to("cuda").eval()
    umm_model.config.interfere_audio = False

    dataset = AudioDataset(root_dir='/mnt/bn/cyz-lq-nas/s3prl_datasets/LibriSpeech', 
                    target_dir='/mnt/bn/cyz-lq-nas/s3prl_gendir/LibriSpeech/fix/yuanzhe_v0.3.2_75k') 
    dataloader = DataLoader(dataset, num_workers=8)

 

    print(ckpt_path)
    eval_out_libritts = evaluate(dataloader, umm_model)
