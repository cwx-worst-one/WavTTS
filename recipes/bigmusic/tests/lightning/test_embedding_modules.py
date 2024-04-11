import os

import torch

current_dir = os.path.dirname(__file__)


def test_get_bestrq_umm_tokens():
    import librosa
    from recipes.umm.requires.model_initializer import init_stage3
    from recipes.bigmusic.lightning.embedding_modules import get_bestrq_umm_tokens

    bestrq_cache_dir = os.path.join(
        os.environ["DUMP_DIR"],
        "ai_music/20240312.symbolic.dump/module_cache/bestrq"
    )
    bestrq_ckpt_path = "hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt"
    input_audio_path = os.path.join(current_dir, "../data/testaudio.30s.wav")

    ## To load the umm model, we need to set "BYTED_RAY_CLUSTER" to
    ## some string other than an empty string
    os.environ["BYTED_RAY_CLUSTER"] = "a"
    model = init_stage3(bestrq_ckpt_path, local_rank=0, cache_dir=bestrq_cache_dir)
    audio, _ = librosa.load(input_audio_path, sr=24000, mono=True)
    audio_tensor = torch.tensor(audio[None, :]).to(model["Stage3"].device)
    tokens = get_bestrq_umm_tokens(model, audio_tensor)
    print(tokens.shape)

    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip