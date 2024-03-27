import torch

from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.requires.model_initializer import init_dualumm

"""
@hanoihantrakul 27 March 2024
Basic examples testing different functionalities in DualUMM like token2mel() decoding.
You should be running this script where a GPU (normally 1xH800) is available. Otherwise
the model won't load and the script won't run.
"""

MODELS_DICT = {
    # @hanoihantrakul: 27 March 2024 This is a 10Hz+10Hz DualUMM v3 model
    "dualumm_v3_10hz+10hz": "hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_dual_master/umm_dual_v3_10hz_corrected_vocal_grad_bert-base-multilingual-uncased_EMAEntropy16384x32/checkpoints/step=0550000.ckpt"
}


def load_model(ckpt_path, cache_dir):
    DUMMY_RANK = 0
    token_model = init_dualumm(ckpt_path, DUMMY_RANK, cache_dir)["Stage3"].eval()
    return token_model


def test_dualumm_methods():
    ckpt_path = MODELS_DICT["dualumm_v3_10hz+10hz"]
    unique_cache_dir = "./.dualumm"

    # Generate test signal
    audio_signal = pitch_utils.gen_test_sin(
        freq=440, amp=0.9, dur_sec=2, sr=24000, add_batch_dim=True
    )
    assert list(audio_signal.shape) == [1, 48000]

    # Load model
    token_model = load_model(ckpt_path, unique_cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract vocal tokens
    tokens_vocal = token_model.wav2token(audio_signal.to(device), wav_type="vocal")
    assert list(tokens_vocal.shape) == [1, 20]
    # Extract instrumental tokens
    tokens_inst = token_model.wav2token(audio_signal.to(device), wav_type="inst")
    assert list(tokens_inst.shape) == [1, 20]

    # Decode vocal tokens to mel
    mel_vocal = token_model.token2mel(tokens_vocal, token_type="vocal")
    assert list(mel_vocal.shape) == [1, 200, 160]
    # Decode instrumental tokens to mel
    mel_inst = token_model.token2mel(tokens_inst, token_type="vocal")
    assert list(mel_inst.shape) == [1, 200, 160]

    # Interleave vocal and instrumental tokens to form full mix tokens
    full_mix_tokens = torch.stack([tokens_vocal, tokens_inst], -1)
    full_mix_tokens = torch.flatten(full_mix_tokens, 1)
    assert list(full_mix_tokens.shape) == [1, 40]  # 20+20=40 tokens
    # Decode full mix tokens to mel
    mel_full_mix = token_model.token2mel(full_mix_tokens, token_type="full")
    assert list(mel_full_mix.shape) == [1, 200, 160]

    print("All Tests Passed")


if __name__ == "__main__":
    test_dualumm_methods()
