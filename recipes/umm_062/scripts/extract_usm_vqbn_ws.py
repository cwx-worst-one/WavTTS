import torch
import torchaudio
from torch.utils.data import DataLoader

assert torch.cuda.is_available()

from tqdm import tqdm
import os

from recipes.umm_062.modules.lit_module_cyz import USMStage3
import torch.nn.functional as F
import numpy as np
import librosa
from torchaudio_augmentations import Compose
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)

torch.set_float32_matmul_precision('high')


@torch.no_grad()
def weighted_sum_vqbn(model, wav):
    dtype = torch.bfloat16
    is_amp = (dtype in [torch.float16, torch.bfloat16])
    input_dict = model.preprocessing(wav)
    with torch.cuda.amp.autocast(enabled=is_amp, dtype=dtype):
        feature = input_dict["fbank"]
        src_mask = input_dict["src_mask"]
        
        front_end_out, backbone_mask, frontend_shape = model.audio_encoder.frontend(feature, src_mask)
        conformers = model.audio_encoder.acoustic_backbone_module
        all_hidden_states = []
        conformer_input = conformers.pos_enc(front_end_out)
        if backbone_mask is None:
            conformer_mask = None
        else:
            conformer_mask = backbone_mask.unsqueeze(1)
        attn_weights = None
        hidden_states_list = []
        for i, layer in enumerate(conformers.encoders):
            if i == model.config.vq_layer_idx:
                with torch.cuda.amp.autocast(enabled=False):
                    vq_inputs = model._weighted_sum(hidden_states_list)
                    before_vq = vq_inputs
                    vq_inputs = model.vq_proj_in(vq_inputs)
                    if model.config.get("vq_type", None) == "FSQ":
                        vq_embs, vq_ids = model.vq(vq_inputs)
                        vq_loss = None
                    elif model.config.get("vq_type", None) == "EMAEntropy":
                        vq_embs, vq_ids, vq_loss = model.vq(vq_inputs, e_scale=1.0 if model.cnt < 30_000 else 0.0)
                    else:
                        vq_embs, vq_ids, vq_loss = model.vq(vq_inputs)
                    
                    vq_inputs = model.vq_proj_out(vq_embs)
                    return {'before_vq': before_vq, 'after_vq': vq_inputs}
            elif i < model.config.vq_layer_idx:
                conformer_input, conformer_mask = layer(
                    [conformer_input, conformer_mask], is_training=True
                )
                hidden_states_list.append(conformer_input[0] if isinstance(conformer_input, (list, tuple)) else conformer_input)




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
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # tokens = self.umm_model.wav2token(audio).cpu().to(torch.int).numpy()
        # 保存为npy格式
        target_path = os.path.join(self.target_dir, os.path.relpath(self.file_list[index], start=self.root_dir)) + '.npy'
        # np.save(target_path, )
        return audio, target_path


@torch.no_grad()
def evaluate(dataloader, usm_model):

    # prepare dir
    for subdir, dirs, files in os.walk(dataloader.dataset.root_dir):
        for file in files:
            dest_subdir = subdir.replace(dataloader.dataset.root_dir, dataloader.dataset.target_dir)
            os.makedirs(dest_subdir, exist_ok=True)

    for i, (audio, target_path) in tqdm(enumerate(dataloader)):
        audio = audio.to("cuda")
        if audio.shape[1] > 16000*60:
            continue

        pad_len = audio.shape[-1] % (usm_model.extra_params.hop_length * 32)
        if pad_len != 0:
            pad_len = usm_model.extra_params.hop_length * 32 - pad_len
        audio = F.pad(audio, (0, pad_len))

        token = usm_model.wav2token(audio, dtype=torch.bfloat16).cpu().to(torch.int).numpy()

        import pdb  
        pdb.set_trace()

        vqbn = weighted_sum_vqbn(usm_model.model, audio)

        features = torch.cat([vqbn['before_vq'], vqbn['after_vq']], 0).cpu().numpy() 
        
        # print(tokens, '   ', target_path)
        np.save(target_path[0], features)
        


if __name__ == "__main__":

    ckpt_path = '/mnt/bn/cyz-lq-nas/model_cache/usm/stage3_v1.6.1.ckpt'
    usm_model = USMStage3.load_from_checkpoint(ckpt_path).eval().cuda()

    # dataset_list = ['LibriSpeech', 'Vox1', 'IEMOCAP']
    dataset_list = ['LibriSpeech', 'IEMOCAP']
    # dataset_list = ['vcc2020']
    # dataset_name = 'IEMOCAP'

    for dataset_name in dataset_list:
        # root_dir = os.path.join('/mnt/bn/cyz-lq-nas/s3prl_datasets', dataset_name)
        # # target_dir = os.path.join('/mnt/bn/cyz-lq-nas/s3prl_gendir/usm_stage3_wordpiece_chroma50_vq16384x32_EMAEntropy_60k_vqbn', dataset_name)
        # # target_dir = os.path.join('/mnt/bn/cyz-lq-nas/s3prl_gendir/pl_asr_2B_ft_test', dataset_name)
        # # target_dir = os.path.join('/mnt/bn/cyz-lq-nas/s3prl_gendir/v1.5_stage3_45k_vqbn', dataset_name)
        # target_dir = os.path.join('/mnt/bn/cyz-lq-nas/s3prl_gendir/stage3_v1.6.1', dataset_name)

        root_dir = os.path.join('/mnt/bn/cyz-lq-nas/test/input')
        target_dir = os.path.join('/mnt/bn/cyz-lq-nas/test/v1.6.0_output')

        dataset = AudioDataset(root_dir=root_dir, target_dir=target_dir) # modify
        dataloader = DataLoader(dataset, num_workers=12)

        print(ckpt_path)
        print(dataset_name)
        evaluate(dataloader, usm_model)

