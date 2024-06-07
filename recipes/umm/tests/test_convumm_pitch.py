import torch

from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.requires.model_initializer import init_stage3_conv1d

"""
@hanoihantrakul 2JUNE2024
Basic example demonstrating how to load ConvUMM Pitch models:
- Supervised Pitch Loss (SPL)
- Perceptual Pitch Loss (PPL) 
- Supervised Pitch Loss + Perceptual Pitch Loss (SPL + PPL)
"""

MODELS_DICT = {
    # @hanoihantrakul: 2JUNE2024 Supervised Pitch Loss (SPL)
    "convumm_spl": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conv_pitch_losses/umm_direct_stage3_conv1d_2255_2375_supervised_pitch_loss_EMAVQ32768x32/checkpoints/step=0500000.ckpt",
    # @hanoihantrakul: 2JUNE2024 Perceptual Pitch Loss (PPL)
    "convumm_ppl": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conv_pitch_losses/umm_direct_stage3_conv1d_2255_2375_perceptual_pitch_loss_EMAVQ32768x32/checkpoints/step=0450000.ckpt",
    # @hanoihantrakul: 2JUNE2024 Supervised + Perceptual Pitch Loss (SPL+PPL)
    "convumm_spl_ppl": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conv_pitch_losses/umm_direct_stage3_conv1d_2255_2375_supervised_and_perceptual_pitch_loss_EMAVQ32768x32/checkpoints/step=0470000.ckpt"
}


def load_model(ckpt_path, cache_dir):
    print(f"Downloading {ckpt_path}")
    DUMMY_RANK = 0
    # Can reuse stage3_conv1d init function
    token_model = init_stage3_conv1d(ckpt_path, DUMMY_RANK, cache_dir)[
        "Stage3Conv1D"
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

    print(f"ConvUMM Pitch succesfully loaded and tested: {ckpt_path}")


def test_convumm_pitchmodel(model_key):
    ckpt_path = MODELS_DICT[model_key]
    unique_cache_dir = f"./.{model_key}"
    run_model(ckpt_path, unique_cache_dir)


if __name__ == "__main__":
    """The longest amount of time is spent downloading the checkpoint. By default, just test the model you are interested in."""
    test_convumm_pitchmodel("convumm_spl")
    test_convumm_pitchmodel("convumm_ppl")
    test_convumm_pitchmodel("convumm_spl_ppl")
