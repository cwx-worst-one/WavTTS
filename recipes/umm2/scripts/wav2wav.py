from recipes.umm2.scripts.test_stage3_wav2tokens_RVQ import ModelLoader, init_model
import os
import torchaudio

def load_msr_audio(audio_path=None, stereo_24k=False):
    if audio_path is None:
        audio_path = "/mnt/bn/music-llm-nas-lq/qinxin/bak/inp071.generated.wav"
        audio_path = "/mnt/hdfs/qinxin.025/testset/token2wav/inp071.generated.wav"

        if not os.path.exists(audio_path):
            os.system("hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/testset/token2wav/inp071.generated.wav inp071.generated.wav")
            audio_path = "inp071.generated.wav"

    audio, sr = torchaudio.load(audio_path)
    # audio = audio[..., int(sr*60):]

    if sr == 24000:
        audio_24k = audio   
    else:
        audio_24k = torchaudio.functional.resample(audio, sr, 24000)
    
    if stereo_24k:
        audio_24k = audio_24k.unsqueeze(0).cuda()   # [1, 2, 24000*sec]
    else:
        audio_24k = audio_24k[0].unsqueeze(0).unsqueeze(1).cuda()       # [1, 1, 24000*sec]

    if sr == 44100:
        audio_441 = audio
    else:
        audio_441 = torchaudio.functional.resample(audio, sr, 44100)
    audio_441 = audio_441.unsqueeze(0).cuda()       # [1, 2, 441000*sec]
    if audio_441.shape[1] == 1:
        audio_441 = audio_441.repeat(1, 2, 1)

    print(audio_24k.shape[-1] / 24000, audio_441.shape[-1] / 44100)
    return audio_24k, audio_441

def load_required_modules(pl_module, local_rank=0):
    pl_module.requires = {}
    for module_name, loader_config in pl_module.required_modules.items():
        if module_name == "pretrained":
            continue
        print(f"loading module {module_name}...")
        if "loader" in loader_config:
            _args = {k: v for k, v in loader_config.items() if k != "loader"}
            loader = loader_config["loader"](**_args)
            pl_module = loader.load_model(pl_module=pl_module)
        else:
            hpath = loader_config['hpath']
            initializer = loader_config['initializer']
            pl_module.requires.update(initializer(hpath, local_rank=local_rank, cache_dir="./"))
    return pl_module

def init_mel_DiT(ckpt_path, model_name=None):
    if model_name:
        cache_dir = f".module_cache/{model_name}"
    else:
        cache_dir = ".module_cache/umm_DiT"
    model = init_model(ckpt_path=ckpt_path, 
                    cache_dir=cache_dir, 
                    device="cuda", 
                    pl_module_string="recipes.umm2.modules.stages.umm_DiT.Stage2DiT")

    pl_module = model["pl_module"].eval()
    pl_module = load_required_modules(pl_module)

    inference_config = {
            "bn_padding": -5,
            "diffusion_nfe": 10,
            "text_cfg_w": 1.6,
            "mem_efficient": True,
            "diffusion_sampler": "ddim",
            "bn_norm_mean": 0,
            "bn_norm_std": 1,
            "diffusion_precision": "bf16",
            "embed_padding": 0,
            "token_embed_chunk": True,
            "token_embed_chunk_size": 60,
            "token_embed_chunk_mode": "even",   # {max, even}
            "inference_R": 4,
        }

    pl_module.model.prepare_inference(inference_config)
    return pl_module


def wav2wav(pl_module, audio_24k, audio_44100, audio_name="reconstruct", target_dir="./"):
    import time
    os.makedirs(target_dir, exist_ok=True)

    target_wav_path = f"{target_dir}/{audio_name}_441_cfg{pl_module.model.text_cfg_w}.wav"
    # if os.path.exists(target_wav_path):
    #     print(f"wav {target_wav_path} already exists, skip")
    #     return


    st = time.time()
    vae = pl_module.model.inference(audio_24k)
    reconstruct_wav_441 = pl_module.sacodec_embs_to_wav(vae, overlap_len=0)
    time_eclapse = time.time() - st
    print([_v.shape for _v in vae] if isinstance(vae, list) else vae.shape, 
          reconstruct_wav_441.shape, audio_44100.shape, time_eclapse, time_eclapse / (audio_44100.shape[-1] / 44100))
    torchaudio.save(target_wav_path, reconstruct_wav_441[0].cpu(), 44100)
    print(f"save to {target_wav_path}")


    vae = pl_module.get_sacodec_embedding(audio_44100)
    reconstruct_wav_441 = pl_module.sacodec_embs_to_wav(vae.mT)
    print(vae.shape, reconstruct_wav_441.shape, audio_44100.shape)
    torchaudio.save(f"{target_dir}/{audio_name}_441_wvae.wav", reconstruct_wav_441[0].cpu(), 44100)


if __name__ == "__main__":
    import argparse
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default=None)
    args = parser.parse_args()
    
    # align mode (mel-sigma-VAE)
    # [cotrain UMM]
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2DiT/encoder_cotrain_dataid8424_noval_adddropout/checkpoints/step=0245000.ckpt"
    # [freeze UMM]
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2DiT/encoder_cotrain_dataid8424_noval_freezeUMM_lr5e5/checkpoints/step=0195000.ckpt"
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2DiT/fixlite_cotrainUMM_melSigmaVAE_align_fused_rmpad_bucket960k/checkpoints/step=0010000.ckpt"

    # align mode (UQ)
    # [cotrain UMM]
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2DiT/encoder_cotrain_dataid8424_UQ25Hz32dim3bit_dropout0p1_fromMelVAE256k_align/checkpoints/step=0180000.ckpt"
    # [freeze UMM]
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2DiT/encoder_cotrain_dataid8424_UQ25Hz32dim3bit_dropout0p1_fromStage2Fixed_align/checkpoints/step=0175000.ckpt"
    
    # align mode (RVQ4)
    # [mono cotrain UMM]
    # model_name = "mono"
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3DiT/encoder_cotrain_RVQ4_14bit_align_dataid8424_noval_cotrainUMM/checkpoints/step=0170000.ckpt"
    # [stereo cotrain UMM]
    model_name = "stereo_64dim"
    ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3DiT/fixlite_cotrainUMM_4RVQ_stereoToken_align_fused_rmpad_bucket480k_h20/checkpoints/step=0170000.ckpt"
    # model_name = "stereo_32dim"
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3DiT/fixlite_cotrainUMM_4RVQ_stereoEnc32dimToken_align_fused_rmpad_bucket480k_h20/checkpoints/step=0170000.ckpt"
    # [freeze UMM]
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3DiT/encoder_cotrain_RVQ4_14bit_align_dataid8424_noval_freezeUMM/checkpoints/step=0175000.ckpt"

    # prefix mode (60000 valid, no sos)
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2DiT/encoder_cotrain_prefix_dataid8424_noval_adddropout/checkpoints/step=0055000.ckpt"
    # prefix mode (120000 valid, add sos, 5e-5 lr, from 50k)
    # ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2DiT/encoder_cotrain_prefix_dataid8424_noval_adddropout_addsos_5e-5_from50k/checkpoints/step=0120000.ckpt"

    pl_module = init_mel_DiT(ckpt_path, model_name)
    if args.audio_path is None:
        # audio_24k, audio_44100 = load_msr_audio()
        audio_24k, audio_44100 = load_msr_audio(stereo_24k=True)
        # audio_24k[:,0] *= 2
        wav2wav(pl_module, audio_24k, audio_44100, audio_name="test", 
                    # target_dir="./saved/UQ_freezeUMM_180k/")
                    # target_dir="./saved/UQ_cotrainUMM_180k/")
                    # target_dir="./saved/RVQ4_freezeUMM_175k/")
                    # target_dir="./saved/RVQ4_cotrainUMM_175k/")
                    # target_dir="./saved/melVAE_freezeUMM_200k/")
                    # target_dir="./saved/melVAE_cotrainUMM_35k/")
                    target_dir="./saved/RVQ4_cotrainUMM_stereo_95k")

    else:
        wav_list = glob.glob(args.audio_path+"/*.wav")
        for wav_path in wav_list:
            audio_24k, audio_44100 = load_msr_audio(wav_path, stereo_24k=True)
            # audio_24k, audio_44100 = load_msr_audio(wav_path)
            wav2wav(pl_module, audio_24k, audio_44100, audio_name=wav_path.split("/")[-1].split(".")[0], 
                    target_dir="./saved/output/RVQ4_cotrainUMM_170k_stereo_4R_64dim/")
                    # target_dir="./saved/output/RVQ4_cotrainUMM_170k_stereo_4R_32dim/")
                    # target_dir="./saved/output/RVQ4_cotrainUMM_170k_mono_4R_32dim/")

    # pl_module.model.text_cfg_w = 1.0
    # vae = pl_module.model.inference(audio_24k)
    # reconstruct_wav_441 = pl_module.sacodec_embs_to_wav(vae)
    # print(vae.shape, reconstruct_wav_441.shape, audio_44100.shape)
    # torchaudio.save("reconstruct_wav_441_cfg1.0.wav", reconstruct_wav_441[0].cpu(), 44100)