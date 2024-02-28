import torch

from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.requires.model_initializer import init_stage3_conv1d

"""
@hanoihantrakul 2/6/2024
Basic example demonstrating how to load UMM Stage3Conv1D model and call tokenize method.
You should be running this script where a GPU (normally 1xH800) is available. Otherwise
the model won't load and the script won't run.
"""

MODELS_DICT = {
    # @hanoihantrakul: 2/6/2024 this is currently the newest model trained on ID 1154. It is a trained model for Zhang Shuo.
    "ummv3": "hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_dual/umm_stage3_pitch_baseline_conv_1D_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0160000.ckpt",
    # @hanoihantrakul: 2/22/2024 These are ConvUMM models trained on SSTK data for Duc Le.
    "sstk_vocab_size_32k": "hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_conv_sstk/umm_stage3_conv1D_sstk_no_ctc_794_EMAVQ32768x32/checkpoints_for_inference/step=0850000.ckpt",
    "sstk_vocab_size_4k": "hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_conv_sstk/umm_stage3_conv1D_sstk_no_ctc_794_EMAVQ4096x32/checkpoints_for_inference/step=0865000.ckpt",
}


def load_model(ckpt_path, cache_dir):
    DUMMY_RANK = 0
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

    print(f"Stage3Conv1D succesfully loaded and tested: {ckpt_path}")


def test_ummv3_model():
    ckpt_path = MODELS_DICT["ummv3"]
    unique_cache_dir = "./.ummv3"
    run_model(ckpt_path, unique_cache_dir)


def test_sstk_model():
    def test_codebook_dims_is_consistent(token_model, codebook_size, codebook_dim):
        print(f"codebook_size: {token_model.model.vq.codebook_size}")
        print(f"codebook_dim: {token_model.model.vq.codebook_dim}")
        assert token_model.model.vq.codebook_size == codebook_size
        assert token_model.model.vq.codebook_dim == codebook_dim

    # Test variation 1: codebook size 32768
    ckpt_path = MODELS_DICT["sstk_vocab_size_32k"]
    unique_cache_dir = "./.sstk_vocab_size_32k"
    run_model(ckpt_path, unique_cache_dir)
    token_model = load_model(ckpt_path, unique_cache_dir)
    test_codebook_dims_is_consistent(token_model, 32768, 32)

    # Test variation 2: codebook size 4096
    ckpt_path = MODELS_DICT["sstk_vocab_size_4k"]
    unique_cache_dir = "./.sstk_vocab_size_4k"
    run_model(ckpt_path, unique_cache_dir)
    token_model = load_model(ckpt_path, unique_cache_dir)
    test_codebook_dims_is_consistent(token_model, 4096, 32)


if __name__ == "__main__":
    """The longest amount of time is spent downloading the checkpoint. By default, just test the model you are interested in."""
    # test_ummv3_model()
    test_sstk_model()
