import os
import uuid
import torch
import pathlib
import torch.nn.functional as F
from typing import Optional, List, Union
from apps.bigmusic.umm.diffusion.lit_modules import (
    DiffusionU2SInfer,
    ChunkInfer,
    ChunkInfer2,
)
from apps.bigtts.umm.diffusion.lit_modules.infer_utils import save_wav
from recipes.umm.requires.model_initializer import ensure_hdfs_ckpt_is_local


def init_diffusion(
    diffusion_config: dict, local_rank=None, cache_dir=None, device=None
):
    if device is None:
        device = torch.device(f"cuda:{local_rank}")
    diffusion_config = diffusion_config.copy()

    if cache_dir is not None:
        os.makedirs(f"{cache_dir}/{local_rank}", exist_ok=True)

    local_path = ensure_hdfs_ckpt_is_local(diffusion_config["diffusion_ckpt_path"], f"{cache_dir}/{local_rank}")
    print(f"Download {diffusion_config['diffusion_ckpt_path']} to {local_path}")
    diffusion_config["diffusion_ckpt_path"] = local_path

    diffusion_model_cls_name = diffusion_config.pop(
        "diffusion_model_cls", "DiffusionU2sInfer"
    )
    diffusion_model_cls = eval(diffusion_model_cls_name)
    diffusion = diffusion_model_cls(**diffusion_config)
    diffusion.setup(0)
    return {"diffusion": diffusion.to(device)}


def token2wav(
    diffusion,
    umm_token: torch.Tensor,
    prompt_wav_path: str = "",
    prompt_wav: Optional[torch.Tensor] = None,
    uttid: str = "",
    scale: Optional[float] = None,
):
    if not os.path.isfile(prompt_wav_path) and prompt_wav != None:
        assert isinstance(prompt_wav, torch.Tensor)
        prompt_wav = prompt_wav.squeeze().cpu().numpy()
        prompt_wav_path = "prompt.wav"
        save_wav(prompt_wav, prompt_wav_path)
    batch = [
        (
            None,
            None,
            prompt_wav_path if prompt_wav_path else None,
            umm_token,
            uttid,
            scale,
        )
    ]
    with torch.no_grad():
        pure_audio_output = diffusion.predict_step(batch, batch_idx=0)
    return pure_audio_output


def token2wav_batch(
    diffusion,
    umm_tokens: Union[List[torch.Tensor], torch.Tensor],
    prompt_wav_paths: Optional[List[str]] = None,
    prompt_wavs: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    uttids: Optional[List[str]] = None,
    scales: Optional[List[float]] = None,
):
    bs = len(umm_tokens)
    if isinstance(umm_tokens, torch.Tensor):
        assert umm_tokens.ndim == 2, f"{umm_tokens.ndim} != 2"
        umm_tokens = umm_tokens.chunk(bs, dim=0)
    batch = []
    assert not (prompt_wav_paths and prompt_wavs)
    if prompt_wav_paths is None:
        prompt_wav_paths = [""] * bs
        if prompt_wavs is not None:
            assert len(prompt_wavs) == bs
            for bidx, prompt_wav in enumerate(prompt_wavs):
                prompt_wav = prompt_wav.squeeze().cpu().numpy()
                prompt_wav_path = f"prompt_{str(uuid.uuid4())}.wav"
                prompt_wav_paths[bidx] = prompt_wav_path
                save_wav(prompt_wav.T, prompt_wav_path, sr=44100)
    assert len(prompt_wav_paths) == bs

    if uttids is None:
        uttids = [""] * bs
    assert len(uttids) == bs, f"{len(uttids)} != {bs}"
    uttids = [str(uttid if uttid else bidx) for bidx, uttid in enumerate(uttids)]

    if scales is None:
        scales = [None] * bs
    assert len(scales) == bs

    batch = [
        (
            None,
            None,
            prompt_wav_paths[bidx],
            umm_tokens[bidx],
            uttids[bidx],
            scales[bidx],
        )
        for bidx in range(bs)
    ]
    with torch.no_grad():
        pure_audio_output = diffusion.predict_step(batch, batch_idx=0)

    if prompt_wavs is not None:
        for temp_prompt_wav_path in prompt_wav_paths:
            pathlib.Path(temp_prompt_wav_path).unlink()
    
    return pure_audio_output


def wav2token(diffusion, syn_wav_path):
    import librosa

    wav, _ = librosa.load(syn_wav_path, sr=24000, mono=True)
    wav = torch.FloatTensor(wav).unsqueeze(0)
    if diffusion.bn_config["wav_norm"]:
        scale = max(0.001, torch.max(torch.abs(wav)))
        wav = wav / scale * 0.95
    else:
        scale = None
    wav = wav.to(diffusion.device)
    if isinstance(diffusion, ChunkInfer):
        syn_wav = wav
    else:
        syn_wav, _ = diffusion.align_wav(
            wav,
            diffusion.token_sample_rate,
            diffusion.mel_config["sampling_rate"],
            diffusion.umm_frame_rate,
            diffusion.mel_frame_rate,
        )
    syn_umm_token = diffusion.wav2token(syn_wav)
    return syn_umm_token, scale


def wav2token_batch(diffusion, syn_wav_paths):
    import librosa

    scales = [] if diffusion.bn_config["wav_norm"] else None
    syn_wavs = []
    syn_wavlens = []
    for syn_wav_path in syn_wav_paths:
        wav, _ = librosa.load(syn_wav_path, sr=24000, mono=True)
        wav = torch.FloatTensor(wav).unsqueeze(0)
        if diffusion.bn_config["wav_norm"]:
            scale = max(0.001, torch.max(torch.abs(wav)))
            wav = wav / scale * 0.95
            scales.append(scale.item())
        syn_wavs.append(wav)
        syn_wavlens.append(wav.shape[-1])
    max_len = max(syn_wavlens)
    syn_wavs = torch.cat(
        [
            F.pad(syn_wav, (0, max_len - syn_wav.shape[-1]), "constant", 0)
            for syn_wav in syn_wavs
        ],
        dim=0,
    )
    syn_wavs = syn_wavs.to(diffusion.device)
    if not isinstance(diffusion, ChunkInfer):
        syn_wavs, _ = diffusion.align_wav(
            syn_wavs,
            diffusion.token_sample_rate,
            diffusion.mel_config["sampling_rate"],
            diffusion.umm_frame_rate,
            diffusion.mel_frame_rate,
        )
    syn_umm_tokens = diffusion.wav2token(syn_wavs)
    return syn_umm_tokens, scales


@torch.no_grad()
def run_diffusion_vocoder(requires, samples, prompt_wav_path="", prompt_wav=None):
    # samples are the UMM tokens
    diffusion = requires["diffusion"]
    output_wav = token2wav(
        diffusion,
        umm_token=samples,
        prompt_wav=prompt_wav,
        prompt_wav_path=prompt_wav_path,
        uttid="test",
    )
    return output_wav


@torch.no_grad()
def run_diffusion_vocoder_batch(
    requires, samples, prompt_wav_paths=None, prompt_wavs=None
):
    # samples are the UMM tokens
    diffusion = requires["diffusion"]
    output_wavs = token2wav_batch(
        diffusion,
        umm_tokens=samples,
        prompt_wavs=prompt_wavs,
        prompt_wav_paths=prompt_wav_paths,
        uttids=None,
        scales=None,
    )
    return output_wavs


if __name__ == "__main__":

    local_rank = 0
    device = f"cuda:0"
    cache_dir = ".module_cache/"

    # no-streaming
    # hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_50hzDualConvV1_40hzSS.yaml"
    # hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_50hzDualConvV3_125hzSS.yaml"

    # hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_50hzDualConv_125hzSS.yaml"
    # or you can download the files from here: hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/weituo/infer_files/voice_condition_valsets.zip
    syn_wav_path = "/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/voice_condition_valsets/slices/male_husky_0_slice1.wav"
    prompt_wav_path = "voice_condition_valsets/conditions_6s/male_husky_0.wav"

    # streaming infer with prompt free model
    hparams_file = "apps/bigmusic/umm/diffusion/conf/infer_generation_25hzConformer_125hzSS_streaming.yaml"
    prompt_wav_path = ""

    from hyperpyyaml import load_hyperpyyaml
    from samantha.utils.hparams import DotDict

    with open(hparams_file, "r", encoding="utf-8") as fin:
        params = load_hyperpyyaml(fin)

    requires = init_diffusion(params["diffusion_config"], local_rank, cache_dir)

    with torch.no_grad():
        umm_token, scale = wav2token(requires["diffusion"], syn_wav_path)
        # generate wav in output_wavs/test.wav
        token2wav(
            diffusion=requires["diffusion"],
            umm_token=umm_token.squeeze(0),
            prompt_wav_path=prompt_wav_path,
            uttid="test",
            scale=scale,
        )
