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



@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def wav2bn(model, wav, L):
    if wav.dim() == 3:
        wav = wav.squeeze(dim=1)
    wav = model.pad_audio(wav.float())
    feature = model.preprocessing(wav)["mel"]
    audio_feature = model.audio_encoder(feature)
    hidden_states = model.encoder_input_dropout(audio_feature)
    position_embeddings = model.embed_positions(hidden_states)
    for i, layer in enumerate(model.encoder_layers):
        if i == L:
            return hidden_states
        hidden_states = layer(
            hidden_states, position_embeddings=position_embeddings
        )
    return hidden_states


@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def wav2allbn(model, wav, use_layer=None):
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
        if use_layer is not None and i not in use_layer:
            continue
        hidden_states_list.append(hidden_states)
    if len(hidden_states.shape) == 3:
        return torch.cat(hidden_states_list, dim=0)
    elif len(hidden_states.shape) == 2:
        return torch.stack(hidden_states_list, dim=0)


@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def wav2bn_stage3(model, wav, L):
    if wav.dim() == 3:
        wav = wav.squeeze(dim=1)
    wav = model.pad_audio(wav.float())
    feature = model.preprocessing(wav)["mel"]
    audio_feature = model.audio_encoder(feature)
    hidden_states = model.encoder_input_dropout(audio_feature)
    position_embeddings = model.embed_positions(hidden_states)
    for i, layer in enumerate(model.encoder_layers):
        if i == L:
            return hidden_states
        if i == model.config.vq_layer_idx:
            hidden_states = model.vq_proj_in(hidden_states)
            vq_embs, vq_ids, vq_loss = model.vq(hidden_states)
            hidden_states = model.vq_proj_out(vq_embs)
        hidden_states = layer(
            hidden_states, position_embeddings=position_embeddings
        )
    return hidden_states


@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def wav2vqbn(model, wav):
    if wav.dim() == 3:
        wav = wav.squeeze(dim=1)
    wav = model.pad_audio(wav.float())
    feature = model.preprocessing(wav)["mel"]
    audio_feature = model.audio_encoder(feature)
    hidden_states = model.encoder_input_dropout(audio_feature)
    position_embeddings = model.embed_positions(hidden_states)
    for i, layer in enumerate(model.encoder_layers):
        if i == model.config.vq_layer_idx:
            hidden_states = model.vq_proj_in(hidden_states)
            vq_embs, vq_ids, vq_loss = model.vq(hidden_states)
            hidden_states = model.vq_proj_out(vq_embs)  # test
            return hidden_states
        hidden_states = layer(
            hidden_states, position_embeddings=position_embeddings
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
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        audio = self.base_transform(audio)
        # tokens = self.umm_model.wav2token(audio).cpu().to(torch.int).numpy()
        # 保存为npy格式
        target_path = os.path.join(self.target_dir, os.path.relpath(self.file_list[index], start=self.root_dir)) + '.npy'
        # np.save(target_path, )
        return audio, target_path



@torch.no_grad()
def evaluate(dataloader, umm_model):

    # prepare dir
    for subdir, dirs, files in os.walk(dataloader.dataset.root_dir):
        for file in files:
            dest_subdir = subdir.replace(dataloader.dataset.root_dir, dataloader.dataset.target_dir)
            os.makedirs(dest_subdir, exist_ok=True)

    for i, (audio, target_path) in tqdm(enumerate(dataloader)):
        audio = audio.to("cuda")
        # bn = wav2bn_stage3(umm_model, audio, L=24).cpu().numpy()  # modify
        bn = wav2allbn(umm_model, audio, use_layer=[11, 17, 23]).cpu().numpy() 
        # bn = wav2vqbn(umm_model, audio).cpu().numpy()  # modify
        # print(tokens, '   ', target_path)
        np.save(target_path[0], bn)
        


if __name__ == "__main__":

    ckpt_path = "/mnt/bn/cyz-lq-nas/model_cache/umm/umm_stage1_v1.2_30k.ckpt"  #modify
    umm_model = Stage3.load_from_checkpoint(ckpt_path).model.to("cuda").eval()  # modify
    umm_model.config.interfere_audio = False

    dataset_list = ['LibriSpeech', 'IEMOCAP']
    # dataset_list = ['vcc2020']
    # dataset_name = 'LibriSpeech'

    for dataset_name in dataset_list:
        root_dir = os.path.join('/mnt/bn/cyz-lq-nas/s3prl_datasets', dataset_name)
        target_dir = os.path.join('/mnt/bn/cyz-lq-nas/s3prl_gendir/umm/umm_stage1_v1.2_30k.ckpt_b11e23i6', dataset_name)
        dataset = AudioDataset(root_dir=root_dir, target_dir=target_dir) # modify
        dataloader = DataLoader(dataset, num_workers=12)
        print(ckpt_path)
        print(dataset_name)
        print(target_dir)
        evaluate(dataloader, umm_model)
