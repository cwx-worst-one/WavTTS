
import os
import torch
from apps.bigmusic.umm.diffusion.lit_modules import DiffusionU2SInfer, ChunkInfer



def init_diffusion(diffusion_config, local_rank=None, cache_dir=None, device=None):
    if device is None:
        device = torch.device(f"cuda:{local_rank}")
    diffusion = (ChunkInfer if diffusion_config.get("token_chunk_size", None) else DiffusionU2SInfer)(**diffusion_config)
    # load umm model
    diffusion.setup(0)
    return { "diffusion": diffusion.to(device) } 



def token2wav(diffusion, umm_token, prompt_wav=None, uttid=""):
    batch = (None, None, prompt_wav, umm_token, uttid) 
    with torch.no_grad():
        pure_audio_output = diffusion.predict_step(batch, batch_idx=0)
    return pure_audio_output


def wav2token(diffusion, syn_wav_path):
    import librosa
    wav, _ = librosa.load(syn_wav_path, sr=24000, mono=True) 
    wav = torch.FloatTensor(wav).unsqueeze(0)
    if diffusion.bn_config['wav_norm']:
        scale = max(0.001, torch.max(torch.abs(wav)).item())
        wav = wav / scale * 0.95
    wav = wav.to(diffusion.device)
    syn_wav = diffusion.align_wav(
        wav, 
        diffusion.mel_config["sampling_rate"], 
        diffusion.umm_frame_rate, 
        diffusion.mel_frame_rate)
    syn_umm_token = diffusion.wav2token(syn_wav)
    return syn_umm_token

@torch.no_grad()
def run_diffusion_vocoder(requires, samples, prompt_wav=None):
    # samples are the UMM tokens
    diffusion = requires['diffusion']
    output_wav = token2wav(diffusion, 
              umm_token=samples,
              prompt_wav=prompt_wav,
              uttid="test")
    return output_wav


if __name__ == "__main__":

    local_rank = 0
    device = f"cuda:0"
    cache_dir = ".module_cache/"

    # no-streaming
    # hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_50hzDualConvV1_40hzSS.yaml" 
    # hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_50hzDualConvV3_125hzSS.yaml"

    # hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_50hzDualConv_125hzSS.yaml"
    # or you can download the files from here: hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/weituo/infer_files/voice_condition_valsets.zip
    syn_wav_path = "voice_condition_valsets/slices/male_husky_0_slice1.wav"
    prompt_wav_path = "voice_condition_valsets/conditions_6s/male_husky_0.wav"

    # streaming infer with prompt free model
    hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_25hzConformer_125hzSS_streaming.yaml" 
    prompt_wav_path = ""


    from hyperpyyaml import load_hyperpyyaml
    from samantha.utils.hparams import DotDict
    with open(hparams_file, "r", encoding="utf-8") as fin:
        params = load_hyperpyyaml(fin)

    requires = init_diffusion(
        params["diffusion_config"], local_rank, cache_dir, 
    )
    
    with torch.no_grad():
        umm_token = wav2token(requires["diffusion"], syn_wav_path)
        # generate wav in output_wavs/test.wav
        token2wav(
            diffusion = requires["diffusion"], 
            umm_token = umm_token.squeeze(0),
            prompt_wav = prompt_wav_path,
            uttid = "test")
