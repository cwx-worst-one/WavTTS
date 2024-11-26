import torch
from recipes.umm.requires.model_initializer import init_stage3
import torch.nn.functional as F
from recipes.umm.utils.ssim import ssim
from torchaudio.functional import resample

def is_a_tensor_and_included_in_loss(x):
    """Helper function for extracting trainable parameters in generator and discriminator"""
    return isinstance(x, torch.Tensor) and x.requires_grad and x.grad_fn is not None

def compute_l1_loss(decoder_output, target, *args, **kwargs):
    # decoder_output : B x T x n_mel
    # target : B x T x n_mel
    l1_loss = F.l1_loss(decoder_output, target, reduction="none")
    weights = weights_nonzero_speech(target)
    l1_loss = (l1_loss * weights).sum() / weights.sum()
    return l1_loss

def compute_ssim_loss(decoder_output, target, *args, **kwargs):
    # decoder_output : B x T x n_mel
    # target : B x T x n_mel
    bias = 6.0
    assert decoder_output.shape == target.shape
    weights = weights_nonzero_speech(target)
    decoder_output = decoder_output[:, None] + bias
    target = target[:, None] + bias
    ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
    ssim_loss = (ssim_loss * weights).sum() / weights.sum()
    return ssim_loss

def compute_LSGAN_loss(predicted, target_value):
    target_tensor = predicted.new_ones(predicted.size()) * target_value
    return F.mse_loss(predicted, target_tensor)

def weights_nonzero_speech(target):
    # target : B x T x mel
    # Assign weight 1.0 to all labels except for padding (id=0).
    dim = target.size(-1)
    return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

class FrozenUMMTokenizer:
    """
    A wrapper around a trained UMM Tokenizer. It will not registered as pl.module. 
    @hanoihantrakul 23OCT2024
    """

    def __init__(self, ckpt_path, local_rank, cache_dir, tokenizer_audio_sample_rate=24000):
        self.ckpt_path = ckpt_path
        self.local_rank = local_rank
        self.cache_dir = cache_dir
        self.tokenizer_audio_sample_rate = tokenizer_audio_sample_rate
        self.wrapped_tokenizer_model = self._load_tokenizer_model(self.ckpt_path, self.local_rank, self.cache_dir)

    def _load_tokenizer_model(self, ckpt_path, local_rank, cache_dir):
        """Load the lit_module and return the underlying model with weights frozen."""
        frozen_tokenizer_module = init_stage3(ckpt_path, local_rank, cache_dir)["Stage3"]
        frozen_tokenizer_module.eval()
        wrapped_tokenizer_model = frozen_tokenizer_module.model
        for p in wrapped_tokenizer_model.parameters():
            p.requires_grad = False
        print("Successfully Loaded Frozen UMM Tokenizer Model from:", ckpt_path)
        return wrapped_tokenizer_model
        
    def get_wav2pre_vq_latents(self, audio, input_audio_sr):
        if self.tokenizer_audio_sample_rate != input_audio_sr:
            #print("audio", audio.shape)
            audio = resample(audio, orig_freq=input_audio_sr, new_freq=self.tokenizer_audio_sample_rate)
            #print("audio resampled", audio.shape)
        return self.wrapped_tokenizer_model.wav2pre_vq_latents(audio)
    
