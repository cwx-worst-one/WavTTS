import contextlib
import os

import torch
import torch.distributed as dist

import samantha.utils.hdfs_helper as hh


def is_local_zero():
    local_rank = os.getenv("LOCAL_RANK", None)
    return local_rank is None or local_rank == "0"


@contextlib.contextmanager
def local_zero_first():
    if not dist.is_initialized():
        yield
    else:
        if not is_local_zero():
            dist.barrier()
        yield
        if is_local_zero():
            dist.barrier()


def ensure_hdfs_ckpt_is_local(target_path, cache_dir):
    """If the ckpt path is on HDFS then download it to a local cache, otherwise use the filepath directly."""
    if target_path.startswith("hdfs://"):
        if cache_dir is None: cache_dir = "/tmp"
        local_path = f"{cache_dir}/{os.path.basename(target_path)}"
        if not os.path.exists(local_path):
            hh.get(target_path, local_path)
            assert os.path.exists(
                local_path
            ), f"Could not retrieve file from {target_path}."
        return local_path
    else:
        return target_path


def load_torch_script_module(module_path, device):
    module = torch.jit.load(module_path, map_location=device).to(device).eval()
    return module


def init_pretrained_bestrq(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        state_dict = torch.load(local_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]

        return {"pretrained_bestrq_state": state_dict}


def init_pretrained(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        state_dict = torch.load(local_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]
        return {"state_dict": state_dict}


def init_pitch_model(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    _ = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        state_dict = torch.load(local_path, map_location=torch.device("cpu"))
        return {"state_dict": state_dict}


def init_rmvpe(hpath, local_rank, cache_dir=None):
    """Load state dict for RMVPE model."""
    return init_pitch_model(hpath, local_rank, cache_dir)


def init_perceptual_pitch_predictor(hpath, local_rank, cache_dir=None):
    """Load state dict for UMM v2 Perceptual Pitch Model."""
    return init_pitch_model(hpath, local_rank, cache_dir)


def init_bestrq_mel_ctc_vq(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import BestRQMelCTC

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        model = BestRQMelCTC.load_from_checkpoint(local_path).to(device).eval()
        return {"BestRQMelCTCVQ": model}


def init_mulan(hpath, local_rank, cache_dir=None):
    from recipes.musiclm.requires.mulan.mulan_infer_g4 import (
        create_mulan_model,
        mulan_inference,
    )

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
            return {
                "mulan": create_mulan_model(local_path, device=device),
                "mulan_infer_fn": mulan_inference,
            }
        else:
            local_path = hpath
            return {
                "mulan": create_mulan_model(local_path, device=device),
                "mulan_infer_fn": mulan_inference,
            }


def init_stage1(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import Stage1

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        model = Stage1.load_from_checkpoint(local_path).to(device).eval()
        return {"Stage1": model}


def init_stage2(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import Stage2

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        model = Stage2.load_from_checkpoint(local_path).to(device).eval()
        return {"Stage2": model}


def init_stage3(hpath, local_rank, cache_dir=None):
    """Init function for standard Stage3 UMM backbone."""
    from recipes.umm.modules.lit_module import Stage3

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        model = Stage3.load_from_checkpoint(local_path, map_location="cpu").to(device).eval()
        return {"Stage3": model}
    
def init_stage3_rvq(hpath, local_rank, cache_dir=None):
    """Init function for standard Stage3 UMM backbone."""
    from recipes.umm.modules.lit_module_mkii_tts_rope import Stage3TTSRopeRVQ

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        model = Stage3TTSRopeRVQ.load_from_checkpoint(local_path, map_location="cpu").to(device).eval()
        return {"Stage3": model}


def init_dualumm(
    hpath,
    local_rank=None,
    cache_dir=None,
    device=None,
    load_required_modules_in_init=False,
):
    from recipes.umm.modules.lit_module_mkii_dual import DualUMMv2 as DualUMM
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    if device is None:
        device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        state_dict = torch.load(local_path, map_location="cpu")
        prefix = "model."
        model_state_dict = {
            k[len(prefix) :]: v
            for k, v in state_dict["state_dict"].items()
            if k[: len(prefix)] == prefix
        }
        state_dict["hyper_parameters"].update(
            load_required_modules_in_init=load_required_modules_in_init
        )
        stage3_module = DualUMM(**state_dict["hyper_parameters"])
        model = stage3_module.model
        model.load_state_dict(model_state_dict)
        model.eval()
        model.to(device)
        return {"Stage3": model}


def init_convumm_gan(
    hpath,
    local_rank=None,
    cache_dir=None,
    device=None,
    load_required_modules_in_init=False,
):
    """
    Init function for ConvUMM-GAN.

    @hanoihantrakul 21MAY2024: ConvUMM-GAN uses a lit_module that is very similar to DualUMM. Thus it is possible
    to load a ConvUMM-GAN model using `init_dualumm()`. However, I created a new init to make the code clearer.
    """
    from recipes.umm.modules.lit_module_convumm_gan import ConvUMMGAN

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    if device is None:
        device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
    state_dict = torch.load(local_path, map_location="cpu")
    prefix = "model."
    model_state_dict = {
        k[len(prefix) :]: v
        for k, v in state_dict["state_dict"].items()
        if k[: len(prefix)] == prefix
    }
    state_dict["hyper_parameters"].update(
        load_required_modules_in_init=load_required_modules_in_init
    )
    lit_module = ConvUMMGAN(**state_dict["hyper_parameters"])
    model = lit_module.model
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    return {
        "Stage3": model
    }  # Using Stage3 naming convention to keep things consistent. Note: convumm_gan does not have a stage3


def init_stage3_dual_voc(hpath, local_rank, cache_dir=None):
    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        voc_ckpt = ensure_hdfs_ckpt_is_local(hpath, cache_dir=cache_dir)
    state_dict = torch.load(voc_ckpt, map_location="cpu")
    prefix = "model_gen."
    model_state_dict = {
        k[len(prefix) :]: v
        for k, v in state_dict["state_dict"].items()
        if k[: len(prefix)] == prefix
    }
    from recipes.umm.modules.vocoder_task import MelGANVocoder

    voc_module = MelGANVocoder(**state_dict["hyper_parameters"], save_hparams=False)
    model = voc_module.model_gen
    model.load_state_dict(model_state_dict)
    model.to(device).eval()
    print(f"Loading vocoder model from {voc_ckpt}")
    return {"mel_vocoder": model}


def init_stage3_mss(hpath, local_rank, cache_dir=None):
    """Init function for Stage3 UMM backbone trained with MSS task."""
    from recipes.umm.modules.lit_module import Stage3MSS

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        model = Stage3MSS.load_from_checkpoint(local_path).to(device).eval()
        """
        @hanoihantrakul 11-25-2023
        This should really be `{"Stage3MSS": model}, but it simplifies experimentation to keep the same name for downstream tasks.
        """
        return {"Stage3": model}


def init_stage3_conv1d(hpath, local_rank, cache_dir=None):
    """Init function for standard Stage3 Conv UMM backbone."""
    from recipes.umm.modules.lit_module import Stage3Conv1D

    # This model replaces Stage3 UMM conformer encoder with Conv1D encoder
    # It also replaces Stage3 UMM Conv2D reconstruction heads with Conv1D reconstruction heads.

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        model = Stage3Conv1D.load_from_checkpoint(local_path, map_location="cpu").to(device).eval()
        return {"Stage3Conv1D": model}


def init_mkii(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import MKIIVQ

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        model = MKIIVQ.load_from_checkpoint(local_path).to(device).eval()
        return {"mkii": model}


def init_soundstream(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        h_ss = f"{hpath}/ss{local_rank}.pt"
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/ss{local_rank}.pt"
            if not os.path.exists(local_path):
                if not hh.get(h_ss, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
        else:
            local_path = h_ss
    return {"ss": load_torch_script_module(local_path, device)}


def init_soundstream_decoder(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        h_ss = f"{hpath}/ss_decoder_{local_rank}.pt"
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/ss_decoder_{local_rank}.pt"
            if not os.path.exists(local_path):
                if not hh.get(h_ss, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
        else:
            local_path = h_ss
    return {"ss_dec": load_torch_script_module(local_path, device)}


def init_sami_tts_api(fe_version, fe_task, cache_dir="/opt/tiger/sami_tts_api/models"):
    from sami_tts_api.sail import download_model

    model_name = f"{fe_task}__{fe_version}.model"
    local_path = f"{cache_dir}/{model_name}"
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    with local_zero_first():
        if not os.path.exists(local_path):
            fe = download_model(fe_task, cache_dir, fe_version)
            assert fe == local_path
    return local_path

    # model_name = f"{fe_task}__{fe_version}.model"
    # hdfs_path = f"hdfs:///home/byte_speech_sv/zongyu.yin/sami_tts_api/models/{model_name}"
    # local_path = f"{cache_dir}/{model_name}"
    # with local_zero_first():
    #     if not os.path.exists(local_path):
    #         if not hh.get(hdfs_path, local_path):
    #             raise ConnectionError(f"Cannot retrieve file from {hdfs_path}.")
    # return local_path


def init_unified_decoder(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import UnifiedDecoder

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        model = UnifiedDecoder.load_from_checkpoint(local_path).to(device).eval()
        return {"unified_decoder": model}


def init_m1_tagging(hpath, local_rank, cache_dir=None):
    from recipes.mi1.models.music_sft import MI1_MusicTaggingMusicSFT

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        model = (
            MI1_MusicTaggingMusicSFT.load_from_checkpoint(local_path, strict=False)
            .to(device)
            .eval()
        )
        return {"m1_tagging": model}


def init_wvae_encoder(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        h_ss = f"{hpath}/wvae_encoder_{local_rank}.pt"
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/wvae_encoder_{local_rank}.pt"
            if not os.path.exists(local_path):
                if not hh.get(h_ss, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
        else:
            local_path = h_ss
    return {"wvae_encoder": load_torch_script_module(local_path, device)}


def init_melae(
    hpath,
    local_rank=None,
    cache_dir=None,
    device=None,
    load_required_modules_in_init=False,
):
    from recipes.music_dit.lit_module.lit_melae import MelAEKL

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    if device is None:
        device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        cache_dir = f'{cache_dir}/{hpath.split("/")[-3]}'
        os.makedirs(cache_dir, exist_ok=True)
        local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
        state_dict = torch.load(local_path, map_location="cpu")
        prefix = "model."
        model_state_dict = {
            k[len(prefix) :]: v
            for k, v in state_dict["state_dict"].items()
            if k[: len(prefix)] == prefix
        }
        state_dict["hyper_parameters"].update(
            load_required_modules_in_init=load_required_modules_in_init
        )
        melae_pl_module = MelAEKL(**state_dict["hyper_parameters"])
        model = melae_pl_module.model
        model.load_state_dict(model_state_dict)
        model.eval()
        model.to(device)
        print(f"| load melae from {local_path}")
        return {"melae": model}

import torchaudio
def load_example_audio(audio_path=None):
    if audio_path is None:
        os.system("hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/testset/token2wav/inp071.generated.wav .")
        audio_path = "inp071.generated.wav"
        # audio_path = "/mnt/hdfs/qinxin.025/testset/token2wav/inp071.generated.wav"

    audio, sr = torchaudio.load(audio_path)
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000)

    audio = audio[0].unsqueeze(0).unsqueeze(1).cuda()   # [B, 1, T]
    print("audio duration", audio.shape[-1] / 24000)

    return audio

if __name__ == "__main__":
    # audio shape: [B, 1, T] (24kHz)
    audio = load_example_audio()

    # ===================================================
    # 25Hz Hanoi baseline: ConformerUMM_Unified_TTS_ROPE
    # ===================================================
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conformer_unified_tts_rope/umm_stage3_unified_tts_rope_bert-base-multilingual-uncased_EMAEntropy32768x32/checkpoints/step=220000.ckpt"
    # model = init_stage3(ckpt_path, 0, cache_dir="./.module_cache/umm/")
    # model = model["Stage3"].eval()

    # output = model.wav2token(audio)
    # for key in output:
    #     print(key, output[key])


    # ===================================================
    # 25Hz rm padding: ConformerUMM_Unified_TTS_ROPE_rmpad
    # ===================================================
    ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/umm_conformer_unified_tts_rope_rmpad/umm_stage3_unified_tts_rope_bert-base-multilingual-uncased/checkpoints/step=220000.ckpt"
    model = init_stage3(ckpt_path, 0, cache_dir="./.module_cache/umm_rmpad/")
    model = model["Stage3"].eval()

    import torch.nn.functional as F
    audio_length = torch.LongTensor([audio.shape[-1]]).to(audio.device)
    pad_audio = F.pad(audio, (0, 24000*10), mode="constant", value=0)
    output = model.wav2token(audio, audio_length)
    output_pad = model.wav2token(pad_audio, audio_length)

    print("without padding", output["vq_ids"].shape, "duration: {}s".format(audio.shape[-1] / 24000))
    vq_id_shape = output["vq_ids"].shape[-1]
    with open("full.txt", "w") as f:
        f.writelines("\n".join(output["vq_ids"][0].cpu().reshape(-1).numpy().astype(str).tolist()))

    print("with 10s padding", output_pad["vq_ids"].shape, "duration: {}s".format(pad_audio.shape[-1] / 24000))
    with open("full_rmpad.txt", "w") as f:
        f.writelines("\n".join(output_pad["vq_ids"][0].cpu().reshape(-1).numpy().astype(str).tolist()))

    locality = torch.sum(output["vq_ids"] == output_pad["vq_ids"][...,:vq_id_shape]) / torch.prod(torch.tensor(output["vq_ids"].shape))
    print("locality", locality.item() * 100, "%")
