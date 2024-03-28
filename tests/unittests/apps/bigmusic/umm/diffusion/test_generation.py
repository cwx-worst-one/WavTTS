import torch
import pytest
import librosa
from hyperpyyaml import load_hyperpyyaml
from apps.bigmusic.umm.diffusion.requires.model_initializer import init_diffusion, wav2token, token2wav


def generation_with_config(
    params,
    syn_wav_path = "voice_condition_valsets/slices_60/male_husky_0_slice1.wav",
    prompt_wav_path = "voice_condition_valsets/conditions_6s/male_husky_0.wav",
    prompt_wav = None,
    uttid = "",
    local_rank = 0,
    cache_dir = ".module_cache/",
):

    requires = init_diffusion(
        params["diffusion_config"], local_rank, cache_dir, 
    )

    if uttid == "":
        diffusion_ckpt_info = params["diffusion_config"]["diffusion_ckpt_path"].split("/")
        uttid = f"{diffusion_ckpt_info[-3]}_{diffusion_ckpt_info[-1]}"
    
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
    syn_wav_path = "voice_condition_valsets/slices_60/male_husky_0_slice1.wav",
    prompt_wav_path = "voice_condition_valsets/conditions_6s/male_husky_0.wav",
    prompt_wav = None,
    uttid = "",
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
    )


@pytest.mark.skip
def test_25hzConformer_125hzSS_streaming_with_wav():
    prompt_wav_path = "voice_condition_valsets/conditions_6s/male_husky_0.wav"
    prompt_wav, _ = librosa.load(prompt_wav_path, sr=24000, mono=True)
    prompt_wav = torch.FloatTensor(prompt_wav).unsqueeze(0)
    generation_with_config_file(
        syn_wav_path = "voice_condition_valsets/slices_60/male_husky_0_slice1.wav",
        hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_25hzConformer_125hzSS_streaming.yaml",
        prompt_wav = prompt_wav,
        prompt_wav_path = "",
    )


@pytest.mark.skip
@pytest.mark.parametrize("hparams_file", [
    "apps/bigmusic/umm/diffusion/conf/infer_generation_25hzConformer_125hzSS.yaml", 
    "apps/bigmusic/umm/diffusion/conf/infer_generation_25hzConformer_125hzSS_streaming.yaml",
    ])
def test_25hzConformer_125hzSS(hparams_file):
    generation_with_config_file(
        syn_wav_path = "./1min_zh_vocal/24k/1min_female_deep_0.wav",
        prompt_wav_path = "",
        hparams_file = hparams_file,
    )
    

@pytest.mark.skip
@pytest.mark.parametrize("hparams_file", [
    "apps/bigmusic/umm/diffusion/conf/infer_generation_50hzDualConvV3_125hzSS.yaml", 
    "apps/bigmusic/umm/diffusion/conf/infer_generation_50hzDualConvV3_125hzSS_streaming.yaml", 
    "apps/bigmusic/umm/diffusion/conf/infer_generation_20hzDualConvV3_125hzSS_streaming.yaml"
    ])
def test_DualConvV3_125hzSS(hparams_file):
    generation_with_config_file(
        hparams_file = hparams_file, 
    )




