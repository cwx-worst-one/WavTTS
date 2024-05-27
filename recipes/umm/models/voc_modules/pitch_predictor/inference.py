from recipes.umm.models.voc_modules.pitch_predictor.model import PitchPredictor
from recipes.umm.utils.mel_utils import torch_wav2spec

class PerceptualPitchPredictor:
    """
    Perceptual Pitch Predictor wrapper to standardize interface like RVMPE pitch model.
    @hanoihantrakul 1/25/2024
    """

    def __init__(self):
        self.wrapped_model = PitchPredictor()
        # This is required to prevent Pytorch Lightning from thinking this model is a trainable module.
        # Without this requires_grad = False parameter you will get a DDPStrategy error.
        for p in self.wrapped_model.parameters():
            p.requires_grad = False
        # All perceptual losses in UMMMv2 are standardized to mel 160 and SR 24000
        self.n_mels_in = 160
        self.sample_rate = 24000
        # @hanoihantrakul 22MAY2024: Use torch_wav2spec() like ConvUMM-1D, ConvUMM-GAN and DualUMM
        self.mel_transform = lambda x: torch_wav2spec(x, num_mels=self.n_mels_in, sample_rate=self.sample_rate)

    def load_and_eval(self, state_dict):
        """Load state dict from .pt file"""
        try:
            self.wrapped_model.load_state_dict(state_dict)
            self.wrapped_model.eval()
            print("Succesfully loaded Perceptual Pitch Predictor!")
        except Exception as e:
            print(e)

    def forward(self, mel_spectrogram):
        """Forward function expects a mel spectrogram when used as part of a tokenizer training pipeline."""
        assert mel_spectrogram.shape[-1] == self.n_mels_in
        self.wrapped_model.to(mel_spectrogram.device)
        return self.wrapped_model.forward(mel_spectrogram)

    def get_hidden_state(self):
        """Return hidden state for computing perceptual loss."""
        return self.wrapped_model.get_hidden_state()
    
    def get_mel_spectrogram(self, audio_waveform):
        """Get Mel Spectrogram using the same settings as during pitch predictor training."""
        if audio_waveform.ndim == 3: # [batch_size, n_channels=1, n_audio_samples]
            audio = audio.squeeze(dim=1).float()
        assert audio_waveform.ndim == 2 # [batch_size, n_audio_samples] no channel dim
        return self.mel_transform(audio_waveform)

    
    def forward_from_audio(self, audio_waveform):
        """Separate function for predicting f0_hz and vuv when module is used outside of tokenizer training pipeline."""
        return self.forward(self.get_mel_spectrogram(audio_waveform))
