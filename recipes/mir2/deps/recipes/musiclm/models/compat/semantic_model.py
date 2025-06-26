# coding=utf-8
import argparse
import os
import time
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
import yaml
from torch_complex.tensor import ComplexTensor


class SSLFrontend(torch.nn.Module):
    def __init__(
        self,
        fs: int = 24000,
        n_fft: int = 1024,
        win_length: int = 600,
        hop_length: int = 240,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
    ):
        super().__init__()

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            window=window,
            normalized=normalized,
            onesided=onesided,
        )

        self.logmel = LogMel(
            fs=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk
        )
        self.n_mels = n_mels

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        onnx_export: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        feat_mask = make_pad_mask(feats_lens).to(feats_lens.device)

        return input_feats, feat_mask


class LogMel(torch.nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk
        )
        self.mel_options = _mel_options
        self.log_base = log_base

        # Note(kamo): The mel matrix of librosa is different from kaldi.
        melmat = librosa.filters.mel(**_mel_options)
        # melmat: (D2, D1) -> (D1, D2)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self, feat: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat.to(feat.device))
        mel_feat = torch.clamp(mel_feat, min=1e-10)

        if self.log_base is None:
            logmel_feat = mel_feat.log()
        elif self.log_base == 2.0:
            logmel_feat = mel_feat.log2()
        elif self.log_base == 10.0:
            logmel_feat = mel_feat.log10()
        else:
            logmel_feat = mel_feat.log() / torch.log(self.log_base)

        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1).to(feat.device), 0.0
            )
        else:
            ilens = feat.new_full(
                [feat.size(0)],
                fill_value=feat.size(1),
                dtype=torch.long,
                device=feat.device,
            )
        return logmel_feat, ilens


class Stft(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STFT forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # NOTE(kamo):
        #   The default behaviour of torch.stft is compatible with librosa.stft
        #   about padding and scaling.
        #   Note that it's different from scipy.signal.stft

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(
                self.win_length, dtype=input.dtype, device=input.device
            )
        else:
            window = None

        # For the compatibility of ARM devices, which do not support
        # torch.stft() due to the lake of MKL.
        stft_kwargs = dict(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            window=window,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=False,
        )
        output = torch.stft(input, **stft_kwargs)

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(
                1, 2
            )

        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad

            olens = (ilens - self.n_fft) // self.hop_length  # + 1
            output = output[:, :-1]
            output.masked_fill_(make_pad_mask(olens, output, 1).to(input.device), 0.0)
        else:
            olens = None

        return output, olens


def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Examples: With only lengths.

    >>> lengths = [5, 3, 2]
    >>> make_non_pad_mask(lengths)
    masks = [[0, 0, 0, 0 ,0],
             [0, 0, 0, 1, 1],
             [0, 0, 1, 1, 1]]
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))
    bs = lengths.size()[0]
    maxlen = lengths.max()
    # if not isinstance(lengths, list):
    #     lengths = lengths.tolist()
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths.cpu()).unsqueeze(-1)

    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def wav_data_gen_raw(wav_lst, batch=32, sample_rate=24000):
    """
    Args:
        wav_lst: List[str] each line should be  "key  abs_path", support hdfs:// too
        batch  : batch size
    Returns:
        a Generator by `yield (uttids, wav_datas, wav_num_sample_points)`
    """
    with open(wav_lst, "r") as fp:
        wav_lst = [line.strip() for line in fp]

    for batch_index in range(0, len(wav_lst), batch):
        batch_wav_lst = wav_lst[batch_index : batch_index + batch]
        batch_data = []
        batch_data_name = []
        batch_data_len = []

        for wav_index, line in enumerate(batch_wav_lst):
            try:
                if len(line.split()) == 1:
                    wav_path = line
                    uttid = os.path.basename(wav_path)[0:-4]
                else:
                    uttid, wav_path = line.split()
                audio, sr = torchaudio.load(wav_path)

                if audio.shape[0] > 1:  # Multi-Channel
                    audio = audio.mean(dim=0, keepdim=True)

                if sr != sample_rate:
                    audio = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=sample_rate
                    )(audio)

                batch_data_len.append(audio.shape[-1])
                batch_data.append(audio[0])
                batch_data_name.append(uttid)
            except Exception as e:
                print("{}: wav reading failed!{}".format(e, batch_wav_lst[wav_index]))

        batch_data = pad_list(batch_data, pad_value=0.0)

        yield (batch_data_name, batch_data, torch.LongTensor(batch_data_len))


# Compute for each data point the closest center
def compute_codes(dataset, centers, device):
    num_points = dataset.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8)
    codes = torch.zeros(num_points, dtype=torch.long, device=device)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers**2, dim=1).view(1, -1)
    inertia = 0
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece**2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
        inertia += distances[range(distances.shape[0]), min_ind].sum()
    return codes, inertia.item()


def w2v_bert_tokenization(frontend, w2v_model, wavs, centers, device):
    with torch.no_grad():
        b, t = wavs.size()
        with torch.autocast(device_type="cuda", enabled=False):
            feats, feat_mask = frontend(
                wavs, torch.LongTensor([t]).repeat([b]).to(device)
            )
        w2v_embeds, _ = w2v_model(feats, feat_mask)
        # kmeans
        b, t, d = w2v_embeds.shape
        dataset = w2v_embeds.view([b * t, d])
        num_points = dataset.size(0)
        # 5e8 should vary depending on the free memory on the GPU
        # Ideally, automatically ;)
        chunk_size = int(5e8)
        codes = torch.zeros(num_points, dtype=torch.long, device=device)
        centers_t = torch.transpose(centers, 0, 1)  # [1024, 1024]
        centers_norms = torch.sum(centers**2, dim=1).view(1, -1)
        inertia = 0
        for i in range(0, num_points, chunk_size):
            begin = i
            end = min(begin + chunk_size, num_points)
            dataset_piece = dataset[begin:end, :]
            dataset_norms = torch.sum(dataset_piece**2, dim=1).view(-1, 1)
            distances = torch.mm(dataset_piece, centers_t)
            distances *= -2.0
            distances += dataset_norms
            distances += centers_norms
            _, min_ind = torch.min(distances, dim=1)
            codes[begin:end] = min_ind
            inertia += distances[range(distances.shape[0]), min_ind].sum()
        codes = codes.view([b, t])
        return codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, required=True, help="modle config file"
    )
    parser.add_argument(
        "--semantic_model", required=True, help="torch script model pth"
    )
    parser.add_argument("--kmeans_centroids", required=True, help="path")

    parser.add_argument("--wav_lst", type=str, required=True, help="")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--sample_rate", type=int, default=24000, help="")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--out_dir", default="exp/semantic_tokens/", help="")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    ###################################
    config_args = yaml.safe_load(open(args.config_file))
    print("Using TorchScript Model")
    frontend_conf = config_args.get("frontend_conf", {})
    print(frontend_conf)
    frontend = SSLFrontend(
        fs=frontend_conf.get("fs", 24000),
        n_fft=frontend_conf.get("n_fft", 1024),
        win_length=frontend_conf.get("win_length", 600),
        hop_length=frontend_conf.get("hop_length", 240),
    )
    encoder_type = config_args.get("encoder")
    print(encoder_type)
    torch.set_num_threads(30)
    torch._C._jit_set_bailout_depth(0)
    semantic_model = torch.jit.load(
        args.semantic_model, map_location=torch.device("cpu")
    )
    semantic_model.to(args.device)
    semantic_model.eval()
    ###
    kmeans_centroids = torch.from_numpy(np.load(args.kmeans_centroids)).to(args.device)

    wav_batch = wav_data_gen_raw(args.wav_lst, args.batch_size, args.sample_rate)

    batch_time_semantic, batch_time_kmeans = [], []
    batch_wav_dur = []
    for batch_idx, batch in enumerate(wav_batch, start=1):
        batch_data_name, batch_data, batch_data_len = batch
        batch_wav_dur.append(
            batch_data_len.max() * len(batch_data_len) / args.sample_rate
        )

        ################################
        tic = time.time()
        feats, feat_mask = frontend(batch_data, batch_data_len)
        semantic, mask = semantic_model(
            feats.to(args.device), feat_mask.to(args.device)
        )
        batch_time_semantic.append(time.time() - tic)

        for i, name in enumerate(batch_data_name):
            length = mask[i].sum()
            feat = semantic[i, :length].detach()
            np.save("%s/%s.npy" % (args.out_dir, name), feat.cpu().numpy())

            tic = time.time()
            codes, _ = compute_codes(feat, kmeans_centroids, args.device)
            batch_time_kmeans.append(time.time() - tic)
            tokens = codes.cpu().tolist()
            if encoder_type == "wav2vec2_conformer_samiasr":
                tokens_orig = tokens
                tokens = (
                    [tokens_orig[0]]
                    + tokens_orig
                    + [tokens_orig[-1]]
                    + [tokens_orig[-1]]
                )
                with open("%s/%s.tokens_orig.txt" % (args.out_dir, name), "w") as f:
                    f.write(" ".join([str(x) for x in tokens_orig]) + "\n")

            with open("%s/%s.tokens.txt" % (args.out_dir, name), "w") as f:
                f.write(" ".join([str(x) for x in tokens]) + "\n")

        if batch_idx % 100 == 0:
            t_semantic, t_kmeans = sum(batch_time_semantic), sum(batch_time_kmeans)
            t_wav = sum(batch_wav_dur)
            print(
                "Finish the %d batch(len %.3fs) in (semantic %.3fs)+(kmeans %.3fs), RTF=%.3f"  # noqa
                % (
                    batch_idx,
                    t_wav,
                    t_semantic,
                    t_kmeans,
                    (t_semantic + t_kmeans) / t_wav,
                )
            )

    t_semantic, t_kmeans = sum(batch_time_semantic), sum(batch_time_kmeans)
    t_wav = sum(batch_wav_dur)
    print(
        "Finish the %d batch(len %.3fs) in (semantic %.3fs)+(kmeans %.3fs), RTF=%.3f"
        % (batch_idx, t_wav, t_semantic, t_kmeans, (t_semantic + t_kmeans) / t_wav)
    )


if __name__ == "__main__":
    main()
