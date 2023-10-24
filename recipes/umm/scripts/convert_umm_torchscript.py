import torch
import os
import numpy as np
import librosa
from tqdm import tqdm
from recipes.umm.requires.model_initializer import init_stage3

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class UMMTokenizer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
    def forward(self, wav):
        # wav: [b, t]
        return self.model.wav2token(wav)
def convert_umm_tokenizer(org_model, wavs, device_id, wav_check_num=1):
    with torch.no_grad():
        print("Converting tokenizer...")
        device = "cuda:{}".format(device_id)
        org_model = org_model.to(device)
        output_pattern = "umm_tokenizer_{}.pt".format(device_id)
        model = UMMTokenizer(org_model.model)
        model = model.to(device).eval()
        # inputs
        # wav = torch.from_numpy(wavs[0]).float().unsqueeze(0).to(device) # [b=1, t]
        wav = torch.randn([1, 24000*600], device=device).float()
        wav = org_model.model.pad_audio(wav)
        hop_length = org_model.model.config.hop_length
        pad_len = wav.shape[-1] % (hop_length * 4)
        if pad_len != 0:
            pad_len = hop_length * 4 - pad_len
        wav = torch.nn.functional.pad(wav, (0, pad_len))
        # script trace
        vq_ids = org_model.model.wav2token(wav)
        print("Org model infer sucess!")
        script_model = torch.jit.trace(model, wav) 
        torch.jit.save(script_model, output_pattern)
        print("Converted!")
        # inference for double check
        diff_token_count = 0
        total_token_count = 0
        total_wav_time_seconds = 0
        script_model = torch.jit.load(output_pattern, map_location=device).eval()
        wav_count = 0
        for wav in tqdm(wavs):
            wav = torch.from_numpy(wav).float().unsqueeze(0).to(device) # [b=1, t]
            total_wav_time_seconds += wav.numel() / 24000
            hop_length = org_model.model.config.hop_length
            pad_len = wav.shape[-1] % (hop_length * 4)
            if pad_len != 0:
                pad_len = hop_length * 4 - pad_len
            wav = torch.nn.functional.pad(wav, (0, pad_len)) #pad 1
            wav = org_model.model.pad_audio(wav) #pad2
            script_outputs = script_model(wav)
            pytorch_outputs = org_model.model.wav2token(wav)
            diff = script_outputs - pytorch_outputs
            diff_token_count += (diff !=0 ).sum().item()
            total_token_count += pytorch_outputs.numel()
            if diff.abs().max().item() > 0 :
                print(diff)

            wav_count = wav_count+1
            if wav_count > wav_check_num:
                break

        print('diff_token_count:', diff_token_count)
        print('total_token_count:', total_token_count)
        print('diff rate:', diff_token_count/total_token_count)
        print('total_wav_time(h):', total_wav_time_seconds / 60 / 60)


ckpt_path = 'step=160000.ckpt'
wav_path = 'Sarah/wav/source/'
wavs = [os.path.join(wav_path, w) for w in os.listdir(wav_path) if w.endswith('.wav')]
wavs = [librosa.load(w, sr=24000)[0] for w in wavs]

for device_id in range(torch.cuda.device_count()):
    org_model = init_stage3(ckpt_path, device_id, './')['Stage3']
    convert_umm_tokenizer(org_model, wavs, device_id=device_id, wav_check_num=1)