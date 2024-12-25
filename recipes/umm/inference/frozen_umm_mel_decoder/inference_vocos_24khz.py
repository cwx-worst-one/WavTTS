import torch
import torchaudio
from recipes.umm.inference.frozen_umm_mel_decoder import inference_helpers

"""
25DEC2024 @hanoihantrakul
Reference script for running inference on FrozenUMM-Mel-Decoder and a pretrained Vocos vocoder

Remember that this model is limited by the fact that the pretrained tokenizer was trained using
100Hz Mel Features while the Vocos mel transform has 93.75Mel Features.

This mis match causes the inverted audio to only sound acceptable for the first ~2 seconds.

```
python3 /opt/tiger/samantha/recipes/umm/inference/frozen_umm_mel_decoder/inference_vocos_24khz.py
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
from recipes.umm.modules.lit_module_umm_mel_decoder import UMMMelDecoderTrainingTaskStereo24khzVocos
"""
24DEC2024 @hanoihantrakul: This model was trained with `vocos.feature_extractors.MelSpectrogramFeatures()`
as the target mel transform.
"""
CKPT_PATH = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_mel_decoder/24khz_stereo_mel_decoder_MonoToStereoDecoding_vocos_truncated_mel/checkpoints/step=0100000.ckpt"
pl_decoder_module = UMMMelDecoderTrainingTaskStereo24khzVocos.load_from_checkpoint(CKPT_PATH)
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

# Prepare VOCOS handler
from recipes.umm.models.umm_mel_decoder_helpers import OpenSourceVocosHandler
vocos_handler = OpenSourceVocosHandler()
vocos_handler.load_opensource_vocos()
vocos_handler.opensource_vocos.to(device)

# Invert to audio with Vocos
audio_pred = vocos_handler.mel2wav_stereo(mel_for_inversion.to(device))
print("audio_pred.shape", audio_pred.shape) # Vocos only works at 24khz

# Save the audio
for i in range(5):
    inference_helpers.save_audio_file(audio_pred[i].cpu().detach(), 
                                    24000, 
                                    f"./audio{i}_wav2prevq2decodermel2wav_24khz_stereo.wav")

""""
Additional Functions:

If you want to inspect the ground truth spectrogram signal do this:
```
ground_truth_mel_spec = vocos_handler.wav2mel(audio, SAMPLE_RATE, zero_pad_missing_length=True) # the `zero_pad_missing_length` makes the 93.75Hz feature zeropadded to expected 100Hz (suboptimal solution)
```


If you want to inspect the quality of the Vocos vocoder on ground truth signals do this:
```
ground_truth_audio_recon = vocos_handler.wav2mel2wav_stereo(audio, SAMPLE_RATE)
```
"""