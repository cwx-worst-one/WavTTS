import io
import pickle
import time

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from samantha.dataio.utils import parquet_reader


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=torch.hann_window(win_size).to(dtype=y.dtype, device=y.device),
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def trim_silence(wav):
    """
    Trim leading and trailing silence
    """
    # These params are separate and tunable per dataset.

    wav = np.pad(wav, (5400, 5400))

    unused_trimed, index = librosa.effects.trim(
        wav, top_db=30, frame_length=512, hop_length=128
    )
    # num_sil_samples = int(8 * 300)
    # head silence is set as half of num_sil_samples
    start_idx = max(index[0] - 3200, 0)
    # tail silence is set as twice of num_sil_samples
    stop_idx = min(index[1] + 5400, len(wav))

    trimmed = wav[start_idx:stop_idx]
    return trimmed


@torch.no_grad()
def process_batch(
    encoder,
    batch,
    samples,
    n_fft=2048,
    sample_rate=24_000,
    hop_length=300,
    win_length=1200,
    device="cuda",
):
    length = [e.shape[-1] for e in batch]
    # max_length = max(length)
    max_length = max(max(length), 249000)  # 10s 每次推理都用同样的size，保证显存cache能复用 max(length)
    batch = [
        F.pad(e, (0, max_length - length[i], 0, 0, 0, 0), value=0.0)
        for i, e in enumerate(batch)
    ]
    batch_wav = torch.cat(batch, dim=0)
    batch_spec = spectrogram_torch(
        batch_wav.squeeze(1), n_fft, sample_rate, hop_length, win_length
    )
    _, batch_m, batch_logs = encoder(batch_wav.to(device), batch_spec.to(device))
    print(batch_wav.shape, batch_spec.shape, batch_m.shape, batch_logs.shape)
    # decoder = torch.jit.load("/mnt/bn/jcong-bn-us/models/wavevae_decoder.pt").to(device).eval() # noqa
    for i, ilen in enumerate(length):
        isample = samples[i]
        m, logs = (
            batch_m[i : i + 1, :, : ilen // 600],
            batch_logs[i : i + 1, :, : ilen // 600],
        )
        m = m[0].permute(1, 0)
        logs = logs[0].permute(1, 0)
        # z_outputs = m + torch.rand_like(m) * torch.exp(logs)
        # de_wav = decoder(z_outputs)[0][0].cpu().numpy()
        # de_wav *= (32767) / max(0.01, np.max(np.abs(de_wav)))
        # uttid = isample["uttid"]

        # write(f"wavs2/de_{uttid}.wav", 24000, de_wav.astype(np.int16))
        # open(f"wavs2/ori_{uttid}.wav", "wb").write(isample["audio"])
        bn = torch.cat([m, logs], -1)
        bn = bn.cpu().numpy()
        item = {"uttid": isample["uttid"], "bns": pickle.dumps(bn)}

        yield item


@torch.no_grad()
def encode(
    encoder,
    parquet_file,
    n_fft=2048,
    sample_rate=24_000,
    hop_length=300,
    win_length=1200,
    device="cuda",
):
    bns = []
    s = time.perf_counter()
    total_audio_dur = 0
    total_cost = 0
    batch_size = 64
    batch, samples = [], []
    for _, sample in tqdm.tqdm(parquet_reader(parquet_file)):
        wav, sr = librosa.load(io.BytesIO(sample["audio"]), sr=None)
        audio_dur = wav.shape[0] / float(sr)
        total_audio_dur += audio_dur
        if len(wav.shape) == 2 and wav.shape[-1] == 2:
            wav = wav[:, 0]
        wav *= 1.0 / max(0.01, np.max(np.abs(wav)))
        if sr != sample_rate:
            print("convert sr ...")
            wav = librosa.core.resample(wav, sr, sample_rate)
        wav = torch.from_numpy(trim_silence(wav)).float()
        wav = torch.stack([wav]).unsqueeze(1).float()
        wav = F.pad(
            wav,
            (0, (wav.size(-1) // 600 + 1) * 600 - wav.size(-1), 0, 0, 0, 0),
            value=0.0,
        )
        batch.append(wav)
        samples.append(sample)
        if len(batch) == batch_size:
            for item in process_batch(
                encoder,
                batch,
                samples,
                n_fft,
                sample_rate,
                hop_length,
                win_length,
                device,
            ):
                bns.append(item)
            batch, samples = [], []

    if batch:
        for item in process_batch(
            encoder, batch, samples, n_fft, sample_rate, hop_length, win_length, device
        ):
            bns.append(item)

    e = time.perf_counter()
    total_cost = e - s
    avg_rtf = total_cost / total_audio_dur
    print(f"{total_audio_dur=:.4f}, {total_cost=:.4f}, {avg_rtf=:.4f}")
    return bns


def main():
    device = "cuda"
    model = (
        torch.jit.load("/mnt/bn/jcong-bn-us/models/wavevae_encoder.pt")
        .to(device)
        .eval()
    )
    bns = encode(
        model,
        "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/bigtts/test_podcast/part=00099/shard=00000.parquet",  # noqa
        device=device,
    )
    print(len(bns))


if __name__ == "__main__":
    main()
