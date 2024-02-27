import argparse
import os
from tqdm import tqdm
import torch
from scipy.io.wavfile import write
from functools import partial
import librosa
from recipes.voicebox.vocoder.BigVGAN.meldataset import mel_spectrogram

def save_wav(audio, output_file, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return

def get_step_epoch_from_ckpt(ckpt_path):
    data = torch.load(ckpt_path)
    global_step = data["global_step"]
    epoch = data["epoch"]
    return global_step, epoch


def prepare_mel_transform():
    #mel_transform = partial(mel_spectrogram,
    #        n_fft=2048,
    #        num_mels=80,
    #        sampling_rate=24000,
    #        hop_size=300,
    #        win_size=1200,
    #        fmin=0, fmax=12000
    #        )
    mel_transform = partial(mel_spectrogram,
            n_fft=4096,
            num_mels=100,
            sampling_rate=24000,
            hop_size=600,
            win_size=2400,
            fmin=0, fmax=12000
            )
    return mel_transform

def prepare_vocoder(args, device):
    vocoder = torch.jit.load(args.vocoder_ckpt_path, map_location=device).eval()
    return vocoder

@torch.no_grad()
def main(args):
    device = args.device
    mel_transform = prepare_mel_transform()
    vocoder = prepare_vocoder(args, device)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    wav_files = os.listdir(args.test_wav_dir)

    for i, wav_file in tqdm(enumerate(wav_files)):
        if not wav_file.endswith("wav"):
            continue

        uttid = wav_file[:-4]

        wav_path = os.path.join(args.test_wav_dir, wav_file)
        wav, _ = librosa.load(wav_path, sr=24000, mono=True) 
        wav = torch.FloatTensor(wav).unsqueeze(0)
        #wav, predict_token, uttid = loaded_data
        crop_wav_len = wav.shape[1] % 4800
        if crop_wav_len > 0:
            wav = wav[:, :-crop_wav_len] 
        scale = max(0.001, torch.max(torch.abs(wav)))
        wav = wav / scale * 0.95
        wav = wav.to(device)

        prompt_mel = mel_transform(wav)
        prompt_wav= vocoder(prompt_mel).squeeze().cpu().numpy()
        output_path = os.path.join(out_dir, uttid+".wav")
        save_wav(prompt_wav, output_path)


if __name__ == "__main__":
    # Generation configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocoder_ckpt_path", type=str, required=True)
    parser.add_argument("--test_wav_dir", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cpu", help='Inference device, "cpu" or "cuda"'
    )
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)