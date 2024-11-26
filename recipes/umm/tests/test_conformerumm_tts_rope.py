import torch

from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.requires.model_initializer import init_stage3

"""
@hanoihantrakul 17JUL2024
Basic example demonstrating how to load ConformerUMM_TTS_ROPE
"""

MODELS_DICT = {
    # @hanoihantrakul: 17JUL2024 for Vocal Music Pipeline
    "conformerumm_tts_rope_vocal": "hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_conformer_2255mixedZHEN_tts_rope/umm_stage3_2255mixedZHEN_2375Speech_tts_rope_bert-base-multilingual-uncased/checkpoints/step=100000.ckpt",
    # @hanoihantrakul: 17JUL2024 for Instrumental Music Pipeline
    "conformerumm_tts_rope_sstk": "hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_conformer_tts_rope_sstk/umm_stage3_sstk_tts_rope_None_EMAEntropy32768x32/checkpoints/step=100000.ckpt",
   }


def load_model(ckpt_path, cache_dir):
    print(f"Downloading {ckpt_path}")
    DUMMY_RANK = 0
    # Can reuse stage3 init function
    token_model = init_stage3(ckpt_path, DUMMY_RANK, cache_dir)[
        "Stage3"
    ].eval()
    return token_model


def run_model(ckpt_path, cache_dir):
    """Load the model and run it on a test sine signal."""
    # Load model
    token_model = load_model(ckpt_path, cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate test signal
    audio_signal = pitch_utils.gen_test_sin(
        freq=440, amp=0.9, dur_sec=15, sr=24000, add_batch_dim=True
    )
    assert list(audio_signal.shape) == [1, 48000]

    # Extract tokens
    audio_tokens = token_model.wav2token(audio_signal.to(device))
    assert list(audio_tokens.shape) == [1, 50]

    print(f"ConformerUMM_TTS_TOPE succesfully loaded and tested: {ckpt_path}")


def test_model(model_key):
    ckpt_path = MODELS_DICT[model_key]
    unique_cache_dir = f"./.{model_key}"
    run_model(ckpt_path, unique_cache_dir)


if __name__ == "__main__":
    """The longest amount of time is spent downloading the checkpoint. By default, just test the model you are interested in."""
    test_model("conformerumm_tts_rope_vocal")
    test_model("conformerumm_tts_rope_sstk")
