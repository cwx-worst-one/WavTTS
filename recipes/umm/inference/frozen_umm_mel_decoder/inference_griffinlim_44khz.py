import torch
import torchaudio
from recipes.umm.inference.frozen_umm_mel_decoder import inference_helpers

"""
25DEC2024 @hanoihantrakul
Reference script for running inference on FrozenUMM-Mel-Decoder and a GriffinLim algorithm.

Remember that the audio quality from GriffinLim will be the worse. This is because
the conversion from Mel to STFT is a nonlinear/nonperfect process. When GriffinLim
attempts to reconstruct phase from this imperfect STFT, there will be many artefacts.

The advantage of GriffinLim at the time of writing is how it works at 44.1khz stereo natively.
All the other vocoders here (MelGAN and Vocos) were trained and limited to mono 24khz.

```
python3 /opt/tiger/samantha/recipes/umm/inference/frozen_umm_mel_decoder/inference_griffinlim_44khz.py
```
"""

assert torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 44100
DURATION_SEC = 30

# Load Audio
audio_list = []
for i in range(5):
    audio_file = f"./recipes/umm/inference/frozen_umm_mel_decoder/sample_{i}.mp3"
    waveform, sample_rate = torchaudio.load(audio_file)
    audio_list.append(waveform)
assert sample_rate == SAMPLE_RATE # the loaded audio is at 44100Hz

# Trim audio to consistent lengths and make mono
audio_mono = inference_helpers.trim_audio_list_to_same_length_and_mono(audio_list, DURATION_SEC, SAMPLE_RATE) # This functions handles sample rate conversion to 24khz
print("audio_mono.shape", audio_mono.shape)

# Load the Frozen-Mel-Decoder
from recipes.umm.modules.lit_module_umm_mel_decoder import UMMMelDecoderTrainingTaskStereoGriffinLim
"""
24DEC2024 @hanoihantrakul: This model was trained with the torchaudio implementation
of a mel spectrogram transform which is compatible with the default torchaudio GriffinLim algorithm.
"""
CKPT_PATH = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_mel_decoder/44khz_stereo_mel_decoder_MonoToStereoDecodingGriffin_l1-1.0_ssim-1.0_adv-0.05_num_conv_layers-8_conv_hidden_size-512hidden_size-2048/checkpoints/step=0215000.ckpt"
pl_decoder_module = UMMMelDecoderTrainingTaskStereoGriffinLim.load_from_checkpoint(CKPT_PATH)
pl_decoder_module.setup("predict")

# Extract the underlying model and tokenizer for manipulation
frozen_tokenizer = pl_decoder_module.frozen_tokenizer
frozen_umm_decoder = pl_decoder_module.model

# Get the prevq_latents
pre_vq_latents = frozen_tokenizer.get_wav2pre_vq_latents(audio_mono.to(device), SAMPLE_RATE)
print("pre_vq_latents.shape", pre_vq_latents.shape)

# Pass prevq_latents to the decoder
mel_out = frozen_umm_decoder.forward(pre_vq_latents)
mel_pred_left, mel_pred_right = torch.tensor_split(mel_out, 2, dim=2) 
print("mel_pred_left.shape", mel_pred_left.shape)
print("mel_pred_right.shape", mel_pred_right.shape)

# Prepare mel spectrogram for inversion
mel_for_inversion = inference_helpers.prepare_mel_for_inversion(mel_pred_left, mel_pred_right)

# Prepare GriffinLim handler
griffin_lim_handler = pl_decoder_module.griffin_lim_handler

# Invert to audio with Vocos
audio_pred = griffin_lim_handler.mel2wav_stereo(mel_for_inversion.to(device))
print("audio_pred.shape", audio_pred.shape) 

# Save the audio
for i in range(5):
    inference_helpers.save_audio_file(audio_pred[i].cpu().detach(), 
                                    SAMPLE_RATE, 
                                    f"./audio{i}_wav2prevq2decodermel2wav_24khz_stereo.wav")

""""
Additional Functions:

If you want to inspect the ground truth spectrogram signal do this:
```
ground_truth_mel_spec = griffin_lim_handler.wav2mel(audio, SAMPLE_RATE)
```


If you want to inspect the quality of the GriffinLim algorithm on ground truth signals do this:
```
ground_truth_audio_recon = griffin_lim_handler.wav2mel2wav_stereo(audio, SAMPLE_RATE)
```
"""