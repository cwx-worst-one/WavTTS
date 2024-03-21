import torch
import pytest
import librosa
from hyperpyyaml import load_hyperpyyaml
from apps.bigmusic.umm.diffusion.requires.model_initializer import init_diffusion, wav2token, token2wav


def generation_with_config(
    params,
    syn_wav_path = "voice_condition_valsets/slices/male_husky_0_slice1.wav",
    prompt_wav_path = "voice_condition_valsets/conditions_6s/male_husky_0.wav",
    prompt_wav = None,
    uttid = "test",
    local_rank = 0,
    cache_dir = ".module_cache/",
):

    requires = init_diffusion(
        params["diffusion_config"], local_rank, cache_dir, 
    )
    
    with torch.no_grad():
        umm_token, scale = wav2token(requires["diffusion"], syn_wav_path)
        # generate wav in output_wavs/test.wav
        token2wav(
            diffusion = requires["diffusion"], 
            umm_token = umm_token.squeeze(0),
            prompt_wav_path = prompt_wav_path,
            prompt_wav = prompt_wav,
            uttid = uttid,
            scale = scale,
        )


def generation_with_config_file(
    hparams_file,
    syn_wav_path = "voice_condition_valsets/slices/male_husky_0_slice1.wav",
    prompt_wav_path = "voice_condition_valsets/conditions_6s/male_husky_0.wav",
    prompt_wav = None,
    uttid = "test",
    local_rank = 0,
    cache_dir = ".module_cache/",
):
    with open(hparams_file, "r", encoding="utf-8") as fin:
        params = load_hyperpyyaml(fin)

    generation_with_config(
        params = params,
        syn_wav_path = syn_wav_path,
        prompt_wav_path = prompt_wav_path,
        prompt_wav = prompt_wav,
        uttid = uttid,
        local_rank = local_rank,
        cache_dir = cache_dir,
    )


@pytest.mark.skip
def test_25hzConformer_125hzSS_streaming_distill():
    generation_with_config_file(
        syn_wav_path = "voice_condition_valsets/slices_60/male_husky_0_slice1.wav",
        hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_25hzConformer_125hzSS_streaming_distill.yaml",
        uttid = "h800_ds1907_prompt_16xH800_25hzUMM_125hzSS_prompt_drop0.1_streaming_chunk1000_0315_consistency_0320_step=90000",
    )


@pytest.mark.skip
def test_25hzConformer_125hzSS_streaming_with_wav():
    prompt_wav_path = "voice_condition_valsets/conditions_6s/male_husky_0.wav"
    prompt_wav, _ = librosa.load(prompt_wav_path, sr=24000, mono=True)
    prompt_wav = torch.FloatTensor(prompt_wav).unsqueeze(0)
    generation_with_config_file(
        syn_wav_path = "voice_condition_valsets/slices_60/male_husky_0_slice1.wav",
        hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_25hzConformer_125hzSS_streaming.yaml",
        uttid = "h800_ds1907_prompt_16xH800_25hzUMM_125hzSS_prompt_drop0.1_streaming_chunk1000_0315_step=550000",
        prompt_wav = prompt_wav,
        prompt_wav_path = "",
    )


@pytest.mark.skip
def test_25hzConformer_125hzSS_streaming():
    generation_with_config_file(
        syn_wav_path = "voice_condition_valsets/slices_60/male_husky_0_slice1.wav",
        hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_25hzConformer_125hzSS_streaming.yaml",
        uttid = "h800_ds1907_prompt_16xH800_25hzUMM_125hzSS_prompt_drop0.1_streaming_chunk1000_0315_step=550000",
    )

@pytest.mark.skip
def test_50hzDualConvV3_125hzSS():
    generation_with_config_file(
        hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_50hzDualConvV3_125hzSS.yaml",
    )




    
