import torch
import torchaudio
from recipes.umm.inference.frozen_umm_mel_decoder import inference_helpers

"""
25DEC2024 @hanoihantrakul
Reference script for running inference on FrozenUMM-Mel-Decoder and a pretrained MelGAN
vocoder.

```
python3 /opt/tiger/samantha/recipes/umm/inference/frozen_umm_mel_decoder/inference_melgan_24khz.py
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
from recipes.umm.modules.lit_module_umm_mel_decoder import UMMMelDecoderTrainingTaskStereo
"""
24DEC2024 @hanoihantrakul: This model was trained with `torch_wav2spec()` as the target mel. It is the same function used to train the MelGAN.
I made a spelling error. Even though the path says 44khz the model is actually trained at 24khz.
"""
CKPT_PATH = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_mel_decoder/44khz_stereo_mel_decoder_MonoToStereoDecoding_l1-1.0_ssim-1.0_adv-0.05_num_conv_layers-8_conv_hidden_size-512hidden_size-2048/checkpoints/step=0415000.ckpt"
pl_decoder_module = UMMMelDecoderTrainingTaskStereo.load_from_checkpoint(CKPT_PATH)
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

# Prepare MELGAN handler
from recipes.umm.models.umm_mel_decoder_helpers import MelGanHandler
melgan_handler = MelGanHandler(local_rank=0,
                               cache_dir=".module_cache")
melgan_handler.load_melgan_weights()

"""
26DEC2024 @hanoihantrakul: 
I did not write the underlying MELGAN system. Unfortunately, due to bad
memory management I can only send one audio file into the loaded MELGAN model.
I am not sure how to fix this and don't have time to do so.

For this part of the script, I had to manually run the script once for every 
audio file, otherwise the script would OOM.
"""
SINGLE_IDX = 0 # 1 # Select only one file for inversion to avoid OOM when sending a full bat
mel_for_inversion_single = mel_for_inversion[SINGLE_IDX:SINGLE_IDX+1, :, :, :]
print("mel_for_inversion_single.shape", mel_for_inversion_single.shape)

# Invert to audio with Melgan
audio_pred = melgan_handler.mel2wav_stereo(mel_for_inversion_single.to(device))
print("audio_pred.shape", audio_pred.shape) # MelGan only works at 24khz

# Save the audio
inference_helpers.save_audio_file(audio_pred[0].cpu().detach(), 
                                  24000, 
                                  f"./audio{SINGLE_IDX}_wav2prevq2decodermel2wav_24khz_stereo.wav")

""""
Additional Functions:

If you want to inspect the ground truth spectrogram signal do this:
```
ground_truth_mel_spec = melgan_handler.wav2mel(audio, SAMPLE_RATE)
```


If you want to inspect the quality of the Melgan vocoder on ground truth signals do this:
```
ground_truth_audio_recon = melgan_handler.wav2mel2wav_stereo(audio, SAMPLE_RATE)
```
"""