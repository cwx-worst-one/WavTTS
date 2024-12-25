import torch
from recipes.umm.requires.model_initializer import init_stage3, init_stage3_dual_voc
from recipes.umm.models.voc_modules.utils import mel2wav as mel2wav_voc_modules
from recipes.umm.utils.mel_utils import torch_wav2spec
import torch.nn.functional as F
from recipes.umm.utils.ssim import ssim
from torchaudio.functional import resample
import torchaudio
import vocos

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
    
class GriffinLimHandler(torch.nn.Module):
    """
    14DEC2024 @hanoihantrakul
    Class for converting audio to mel spectrogram and inverting back to audio using GriffinLim.
    There are settings (e.g. n_stft) that have to be held constant during transformation to mel and
    inversion back to using GriffinLim. This is encapsulated in this class.

    Previous functions like `torch_wav2spec()` or `SpeechTransform` are not configuerd in a way 
    which is compatible with `torchaudio.transforms.GriffinLim`. So I have to train a FrozenUMM-Mel-Decoder
    with a mel transform paired with the inverse GriffinLim transform.
    """
    
    def __init__(self, 
                 sample_rate=44100,
                 data_sample_rate=44100,
                 n_fft=4096,
                 win_length=4096,
                 hop_length=441,
                 n_mels=160,
                 n_channels=2,
                 power=1.0):
        super().__init__()
        self.sample_rate = sample_rate # the operating sample rate e.g. can be 24khz, 32khz, 44.1khz or 48khz
        self.data_sample_rate = data_sample_rate # the sample rate of the original data, usually 44.1khz in FrozenUMM-Mel-Decoder experiments
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length # This hop length has to divide the sample rate to produce 100hz equal to the original tokenizer mel frame rate e.g. 24000/240 = 100, 44100/441 = 100Hz
        self.n_mels = n_mels
        self.n_channels = n_channels
        self.power = power # Whether to convert STFT to a power spectrum (power=2.0). I found the spectrum to look better using power = 1.0
        
        # Since transforms.Spectrogram is also an nn.Module, this whole object needs to be an nn.Module in order to register properly on the GPU
        # See https://discuss.pytorch.org/t/unable-to-move-window-to-gpu-for-torchaudio-spectrogram/163412
        self.stft_transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft,
                                                                win_length=self.win_length,
                                                                hop_length=self.hop_length,
                                                                power=self.power)
        
        # Refer to https://pytorch.org/audio/main/generated/torchaudio.transforms.MelScale.html
        self.mel_transform = torchaudio.transforms.MelScale(sample_rate=self.sample_rate, 
                                                            n_mels=self.n_mels,
                                                            n_stft=self.n_fft // 2 + 1 # due to stft properties you need it to be this number exactly for forward mel transform
                                                           )
        
        # Refer to https://pytorch.org/audio/stable/generated/torchaudio.transforms.InverseMelScale.html
        self.mel_inv_transform = torchaudio.transforms.InverseMelScale(sample_rate=self.sample_rate,
                                                                       n_mels=self.n_mels,
                                                                       n_stft=n_fft // 2 + 1 # due to stft properties you need it to be this number exactly for inverse mel transform
                                                                      )
                                                            
        
        # Refer to https://pytorch.org/audio/main/generated/torchaudio.transforms.GriffinLim.html
        self.griffin_lim_inv_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft,
                                                                          n_iter=32,
                                                                          win_length=self.win_length,
                                                                          hop_length=self.hop_length,
                                                                          power=self.power)
        
        self.resampler = torchaudio.transforms.Resample(orig_freq=self.data_sample_rate, new_freq=self.sample_rate)
        
    def check_and_resample_audio(self, x):
        if self.sample_rate != self.data_sample_rate:
            x = self.resampler(x)
        return x
    
    def check_audio_dims(self, x):
        """ Expects (batch_size, n_channels, num_audio_samples) """
        assert x.shape[1] == self.n_channels
        
    def check_mel_dims(self, x_mel):
        """ Expects (batch_size, n_channels, n_mels, n_mel_timesteps) """
        assert x_mel.shape[1] == self.n_channels
        assert x_mel.shape[2] == self.n_mels
        
    def wav2stft(self, x):
        self.check_audio_dims(x)
        x = self.check_and_resample_audio(x)
        x_stft = self.stft_transform(x)
        return x_stft
    
    def wav2mel(self, x):
        self.check_audio_dims(x)
        x = self.check_and_resample_audio(x)
        x_stft = self.stft_transform(x)
        x_mel = self.mel_transform(x_stft)
        self.check_mel_dims(x_mel)
        return x_mel
    
    def wav2stft2wav_stereo(self, x):
        """
        Convenience function for listening to Griffin Lim quality on ground truth samples using a traditional stft without FrozenUMM encoding/decoding.
        """
        x_stft = self.wav2stft(x)
        x_audio_griffin_lim = self.griffin_lim_inv_transform(x_stft)
        return x_audio_griffin_lim
    
    def wav2mel2wav_stereo(self, x):
        """
        Convenience function for listening to Griffin Lim quality on ground truth samples using a mel spectrogram without FrozenUMM encoding/decoding.
        """
        x_mel = self.wav2mel(x)
        x_stft = self.mel_inv_transform(x_mel)
        x_audio_griffin_lim = self.griffin_lim_inv_transform(x_stft)
        return x_audio_griffin_lim
    
    def mel2wav_stereo(self, x_mel):
        """
        Reconstruct audio waveform from the mel_spectrogram which is normally provided by the FrozenUMM-Mel-Decoder.
        """
        self.check_mel_dims(x_mel)
        x_stft = self.mel_inv_transform(x_mel)
        x_audio_griffin_lim = self.griffin_lim_inv_transform(x_stft)
        return x_audio_griffin_lim

class OpenSourceVocosHandler(torch.nn.Module):
    """
    19DEC2024 @hanoihantrakul
    Class for converting audio to mel spectrogram and inverting back to audio using Vocos. This ensures
    the mel transform used here is compatible with the open source Vocos implementation.

    Note how the Open Souce Vocos implementation uses a hop_size=256 which at 24khz creates an 
    effective mel_feature_rate=93.75Hz which is different from our default 100hz mel_feature_rate.

    There are many ways to solve this problem, including retraining Vocos with a hop_size=240 so that
    mel_feature_rate=100hz as expected. However at the time of writing, I don't have the time to train
    a new Vocos model so I have to use the open source one as-is. 

    Whilst it is possible to do interpolation and make a (16, 375, 128) mel spectogram resampled to
    (16, 400, 128) where [batch_size, time_steps, n_mels]. However, this risks introducing artifacts
    into the mel spectrogram which will be visible in the audio.

    Instead I think the best way to do this is just to zero_pad the 93.75Hz mel_feature sequence and task
    the decoder to decode to a "slightly truncated mel". This way we know the open source Vocos implementation
    will still receive a spectrogram with the statistics it expects. 
    
    24DEC2024 @hanoihantrakul
    In practise, this causes only the first ~2 seconds of generated audio to sound good, the rest of the audio
    sounds blurry because of the inherent mismatch between 93.75Hz and 100Hz. Conclusion: please train
    a new Vocos vocoder at 100Hz. 
    """
    def __init__(self, 
                sample_rate=24000,
                load_vocos=False):
        super().__init__()
        self.vocos_mel_transform = vocos.feature_extractors.MelSpectrogramFeatures()
        self.sample_rate = sample_rate
        assert self.sample_rate == 24000 # this implementation only supports 24khz 
        
        self.target_mel_feature_rate = 100 # this is hardcoded according to the tokenizer mel feature frame rate
        self.n_mels = 100 # this is hardcoded to the vocos open source configuration
        
        self.opensource_vocos = None
        if load_vocos:
            self.load_opensource_vocos()
            
    def load_opensource_vocos(self):
        from vocos import Vocos
        self.opensource_vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    def check_audio_dims(self, x):
        """ Expects (batch_size, n_channels, num_audio_samples) """
        assert x.shape[1] == 2 # although Vocos is mono-only, this wrapper enforces stereo in and stereo out for compatability with the lit_module
        
    def check_mel_dims(self, x_mel):
        """ Expects (batch_size, n_channels, n_mels, n_mel_timesteps) """
        assert x_mel.shape[2] == self.n_mels

    def pad_mel_to_target_rate(self, x_mel, audio_duration_sec):
        """
        As mentioned above, the Vocos mel transform is hard coded to hop_size=256 and not hop_size=240.
        This means the mel features from Vocos are actually 93.75Hz and not 100Hz like we want at 24khz. 

        To make training work, I have to pad these mel features to 100Hz.

        This means the prevq latents are being decoded to a "slightly truncated mel spectrogram"
        which is acceptable for now.         
        """
        original_mel_timesteps = x_mel.shape[-1]
        target_mel_timesteps = audio_duration_sec * self.target_mel_feature_rate
        padding_timesteps = int(target_mel_timesteps - original_mel_timesteps)
        padding = (0, padding_timesteps)
        x_mel_padded = torch.nn.functional.pad(x_mel, padding, 'constant', 0)
        return x_mel_padded

    def wav2mel(self, x, input_audio_sr, zero_pad_missing_length=False):
        """
        Convert audio mel spectrogram using Vocos mel transform (feature_rate=93.75hz)
        """
        if self.sample_rate != input_audio_sr:
            x = resample(x, orig_freq=input_audio_sr, new_freq=self.sample_rate)
        self.check_audio_dims(x)
        x_mel = self.vocos_mel_transform(x)
        if zero_pad_missing_length:
            """
            Initially, my solution for now is to just zeropad to expected length had it been 100hz.

            E.g. (16, 1, 100, 2813) will be zeropadded to (16, 1, 100, 3000). Obviously, in the future
            the Vocos model should be retrained completely to be at the same frate rate.
            """
            audio_duration_sec = x.shape[-1] / self.sample_rate
            x_mel = self.pad_mel_to_target_rate(x_mel, audio_duration_sec)
        self.check_mel_dims(x_mel)
        return x_mel

    def mel2wav_mono(self, x_mel):
        """
        Open Source Vocos can only decode one mono mel spectrogram.
        It expects (batch-size, n_mels, n_timesteps)

        ```
        import torch
        from vocos import Vocos
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        mel = torch.randn(1, 100, 256)  # B, C, T
        audio = vocos.decode(mel)
        """
        assert self.opensource_vocos is not None
        assert x_mel.ndim == 3
        assert x_mel.shape[1] == self.n_mels
        return self.opensource_vocos.decode(x_mel)

    def mel2wav_stereo(self, x_mel):
        """
        Use Vocos to generate stereo audio by separately inverting the left channel then right channel.
        
        Because of the padding to make opensource vocos 93.75Hz feature rate compatible with BigMusic Tokenizer 100hz,
        the output audio will be longer than expected. This part can be ignored.
        
        e.g. expected audio len (16,2,720000) for 30 seconds of audio at 24khz
        in reality, this function will output (16,2,767744). The extra samples of audio can be ignored.
        """
        self.check_mel_dims(x_mel)
        # decode left channel separately
        x_mel_left = x_mel[:, 0, :, :]
        x_audio_left = self.mel2wav_mono(x_mel_left)
        # decode right channel separately
        x_mel_right = x_mel[:, 1, :, :]
        x_audio_right = self.mel2wav_mono(x_mel_right)
        # stack into stereo signal
        x_audio_stereo = torch.cat([x_audio_left[:, None, :], x_audio_right[:, None, :]], axis=1)
        self.check_audio_dims(x_audio_stereo)
        return x_audio_stereo
    
    def wav2mel2wav_stereo(self, x, input_audio_sr):
        """
        Convenience function to check the reconstruction of vocos on ground truth audio. 
        """
        x_mel = self.wav2mel(x, input_audio_sr)
        x_audio_recon = self.mel2wav_stereo(x_mel)
        return x_audio_recon
    
class MelGanHandler(torch.nn.Module):
    """
    24DEC2024 @hanoihantrakul
    Class for converting audio to mel spectrogram and inverting back to audio using a 
    pretrained Mel-GAN Vocoder trained by Ren Yi in ~FEB2024 for DualUMM development.

    The reason I include this hear is because the open source Vocos vocoder is clocked to
    93.75hz feature rate which leads to timing mismatches with our 100hz tokenizer.

    To make sure it is a mis match of time frame issue, I am running the MelGAN vocoder 
    on spectrograms generated by the FrozenUMM-Mel-Decoder to get a sense of sound quality.
    """

    def __init__(self, 
                local_rank,
                cache_dir,
                sample_rate=24000,
                load_melgan=False):
        super().__init__()
        self.sample_rate = sample_rate
        assert self.sample_rate == 24000 # this implementation only supports 24khz 
        '''
        26DEC2024 @hanoihantrakul
        The mel transform used in Ren Yi's work is defined here:
        https://code.byted.org/seed/samantha/blob/master/recipes/umm/modules/vocoder_task.py#L66

        There are confusingly many implementations of the mel transform in the samantha library. 
        '''
        self.n_mels = 160 # hardcoded to 160 in Ren Yi's work
        self.melgan_mel_transform = lambda x: torch_wav2spec(x, num_mels=160, sample_rate=self.sample_rate) # `torch_wav2spec` hardcodes assumption of 1 channel audio input (batch_size, 1, timesteps)
        self.melgan_ckpt = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/renyi/samantha_ckpts/voc/v1/checkpoints/step=4170000.ckpt" # there is only 1 vocoder that was trained
        self.melgan_vocoder = None 
        if load_melgan:
            self.load_melgan_weights()

        self.local_rank = local_rank
        self.cache_dir = cache_dir

    def load_melgan_weights(self):
        vocoder = init_stage3_dual_voc(self.melgan_ckpt, self.local_rank, self.cache_dir)["mel_vocoder"].eval()
        self.melgan_vocoder = vocoder
        
    def check_audio_dims(self, x):
        """ Expects (batch_size, n_channels, num_audio_samples) """
        assert x.shape[1] == 2 # although melgan is mono-only, this wrapper enforces stereo in and stereo out for compatability with the lit_module
        
    def check_mel_dims(self, x_mel):
        """ Expects (batch_size, n_channels, n_mels, n_mel_timesteps) """
        assert x_mel.shape[2] == self.n_mels 

    def wav2mel(self, x, input_audio_sr):
        """
        Convert audio mel spectrogram using MelGAN mel transform.
        """
        if self.sample_rate != input_audio_sr:
            x = resample(x, orig_freq=input_audio_sr, new_freq=self.sample_rate)
        self.check_audio_dims(x)
        x_mel_left = self.melgan_mel_transform(x[:,0,:]) # x_mel_left (batch_size, n_mel_timesteps, n_mels)
        x_mel_left = x_mel_left.transpose(1,2) # x_mel_left (batch_size, n_mels, n_mel_timesteps)
        x_mel_right = self.melgan_mel_transform(x[:,1,:]) # x_mel_right (batch_size, n_mel_timesteps, n_mels)
        x_mel_right = x_mel_right.transpose(1,2) # x_mel_right (batch_size, n_mels, n_mel_timesteps)
        x_mel = torch.concat([x_mel_left[:, None, :, :], x_mel_right[:, None, :, :]], dim=1) # add a channel dimension and concat on channel dimension
        self.check_mel_dims(x_mel)
        return x_mel

    def mel2wav_mono(self, mel):
        """
        24DEC2024 @hanoihantrakul
        The underlying MelGAN vocoder is a mono 24khz model. Unfortunately, the associated inference functions
        were very confusing and only supported single audio inputs instead of batched audio input. I had to 
        trial and error to get the right dimensions. I fixed this problem using a for loop.
        """
        mel_batch_size = mel.shape[0]
        wav_list = []
        for i in range(mel_batch_size):
            wav_out = mel2wav_voc_modules(mel[i], None, self.melgan_vocoder) # I did not write this function and do not agree with the interface. It only accepts single audio inputs, not batched input (???)
            wav_list.append(wav_out[None, :]) # add batch dimension to wav_out (1, num_audio_samples)
        wav = torch.cat(wav_list, dim=0)
        print(wav.shape)
        return wav

    def mel2wav_stereo(self, x_mel):
        """
        This function mainly handles getting the expected dimensions of the mel spectrogram
        matched with the `torch_wav2spec()` function used in the MelGan implementation
        """
        self.check_mel_dims(x_mel)
        '''
        Ufortunately due to bad memory management, the melgan vocoder only supports single audio inputs. 
        It does not support batch sizes larger than 1. I had to try and error this function to get the right dimensions.
        You will need to do a for loop and call this function within the for loop.
        '''
        assert x_mel.shape[0] == 1 # force batch size to be 1 (unoptimal)

        # decode left channel separately
        x_mel_left = x_mel[:, 0, :, :] # x_mel_left (batch_size, n_mel_timesteps, n_mels) 
        x_mel_left = x_mel_left.transpose(1,2) # x_mel_left (batch_size, n_mels, n_mel_timesteps) 
        x_audio_left = self.mel2wav_mono(x_mel_left)
        # decode right channel separately
        x_mel_right = x_mel[:, 1, :, :] # x_mel_right (batch_size, n_mel_timesteps, n_mels) 
        x_mel_right = x_mel_right.transpose(1,2) # x_mel_right (batch_size, n_mels, n_mel_timesteps) 
        x_audio_right = self.mel2wav_mono(x_mel_right)
        # stack left and right channels
        x_audio = torch.concat([x_audio_left[:, None, :], x_audio_right[:, None, :]], dim=1) # add a channel dimension and concat on channel dimension
        self.check_audio_dims(x_audio)
        return x_audio 
        
    def wav2mel2wav_stereo(self, x, input_audio_sr):
        """
        Convenience function to check the reconstruction of MelGan on ground truth audio. 
        """
        x_mel = self.wav2mel(x, input_audio_sr)
        x_audio_recon = self.mel2wav_stereo(x_mel)
        return x_audio_recon