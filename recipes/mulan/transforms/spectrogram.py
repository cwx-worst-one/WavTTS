import torch
import torchaudio

from recipes.mulan.transforms.inverse import InverseTransform


class SpectrogramTransform(InverseTransform):
    def __init__(
        self, n_fft, win_length: int, hop_length: int, window_fn: torch.hann_window
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn

        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=self.window_fn,
            power=None,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_iter=32,
            window_fn=self.window_fn,
            power=1.0,
            momentum=0.99,
            rand_init=True,
        )

    def _spectrogram(self, x):
        return self.spectrogram_transform(x).abs()

    def inverse(self, x):
        return self.griffin_lim(x)

    def forward(self, x):
        return self._spectrogram(x)


class MelSpectrogramTransform(SpectrogramTransform):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        f_min: int,
        f_max: int,
        max_mel_iters: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.max_mel_iters = max_mel_iters
        self.mel_scaler = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=sample_rate,
            n_stft=self.n_fft // 2 + 1,
            f_min=self.f_min,
            f_max=self.f_max,
        )

        # https://github.com/riffusion/riffusion/blob/0610a45e80101fc010e9b2df63077b6095164d39/riffusion/spectrogram_params.py#L33
        self.inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            n_stft=self.n_fft // 2 + 1,
            max_iter=self.max_mel_iters,
            f_min=self.f_min,
            f_max=self.f_max,
        )

    def inverse(self, mel_spec: torch.Tensor) -> torch.Tensor:
        spec = self.inverse_mel_scaler(mel_spec)
        return self.griffin_lim(spec)

    def forward(self, x):
        x = self._spectrogram(x)
        return self.mel_scaler(x)
