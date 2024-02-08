import torch

from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.requires.model_initializer import init_stage3_conv1d

"""
Basic example demonstrating how to load UMM Stage3Conv1D model and call tokenize method.
@hanoihantrakul 2/6/2024
"""


def run_model():
    # @hanoihantrakul: 2/6/2024 this is currently the newest model trained on ID 1154. It is a trained model for Zhang Shuo.
    #CKPT_PATH = "hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_dual/umm_stage3_pitch_baseline_conv_1D_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0160000.ckpt"

    # @hanoihantrakul: 2/8/2024 this is a dummy model trained on SSTK ID 794. It was created to verify loading and inference works as expected for Duc Le.
    CKPT_PATH = "hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_conv_sstk/umm_stage3_conv1D_sstk_no_ctc_794_EMAVQ32768x32_dummy/checkpoints/step=0035000.ckpt"

    # Configure dirs
    CACHE_DIR = "./.test_umm_conv1d_cache"
    DUMMY_RANK = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    token_model = init_stage3_conv1d(CKPT_PATH, DUMMY_RANK, CACHE_DIR)[
        "Stage3Conv1D"
    ].eval()

    # Generate test signal
    audio_signal = pitch_utils.gen_test_sin(
        freq=440, amp=0.9, dur_sec=15, sr=24000, add_batch_dim=True
    )
    assert list(audio_signal.shape) == [1, 48000]

    # Extract tokens
    audio_tokens = token_model.wav2token(audio_signal.to(device))
    assert list(audio_tokens.shape) == [1, 50]

    print("Stage3Conv1D Succesfully loaded and tested!")


if __name__ == "__main__":
    run_model()
