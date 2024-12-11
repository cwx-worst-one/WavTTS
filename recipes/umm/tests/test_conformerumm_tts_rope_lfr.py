import torch

from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.requires.model_initializer import init_stage3

"""
@hanoihantrakul 25NOV2024
Basic example demonstrating how to load ConformerUMM_TTS_ROPE_LFR for low frame rate ablation of 10, 15, 20hz
"""

MODELS_DICT = {
    # @hanoihantrakul: 25NOV2024 15Hz 32768x32 Tokenizer. The step count might seem low but this was trained for 6 days on 4x8 GPU's
    "conformerumm_tts_rope_lfr_15hz": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conformer_tts_rope_lfr/umm_stage3_tts_rope_lfr-15_bert-base-multilingual-uncased_EMAEntropy32768x32/checkpoints/step=080000.ckpt",
    # @hanoihantrakul: 25NOV2024 15Hz 54400x40 Tokenizer. The step count might seem low but this was trained for 6 days on 4x8 GPU's
    "conformerumm_tts_rope_lfr_15hz_larger_codebook": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conformer_tts_rope_lfr/umm_stage3_tts_rope_larger_codebook_lfr-15_bert-base-multilingual-uncased_EMAEntropy54400x48/checkpoints/step=080000.ckpt",
    # @hanoihantrakul: 7DEC2024 20Hz 32768x32 Tokenizer. 
    "conformerumm_tts_rope_lfr_20hz": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conformer_tts_rope_lfr/umm_stage3_tts_rope_lfr-20_bert-base-multilingual-uncased_EMAEntropy32768x32/checkpoints/step=520000.ckpt",
    # @hanoihantrakul: 7DEC2024 20Hz 39208x40 Tokenizer.  
    "conformerumm_tts_rope_lfr_20hz_larger_codebook": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conformer_tts_rope_lfr/umm_stage3_tts_rope_larger_codebook_lfr-20_bert-base-multilingual-uncased_EMAEntropy39208x40/checkpoints/step=310000.ckpt",
   }


def load_model(ckpt_path, cache_dir):
    print(f"Downloading {ckpt_path}")
    DUMMY_RANK = 0
    # Can reuse stage3 init function
    token_model = init_stage3(ckpt_path, DUMMY_RANK, cache_dir)[
        "Stage3"
    ].eval()
    return token_model


def run_model(ckpt_path, cache_dir, expected_frame_rate):
    """Load the model and run it on a test sine signal."""
    # Load model
    token_model = load_model(ckpt_path, cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check frame rate
    assert token_model.model.config.frame_rate == expected_frame_rate

    # Generate test signal
    audio_duration_sec = 2
    sample_rate = 24000
    audio_signal = pitch_utils.gen_test_sin(
        freq=440, amp=0.9, dur_sec=audio_duration_sec, sr=sample_rate, add_batch_dim=True
    )
    expected_audio_samples = int(sample_rate * audio_duration_sec)
    assert list(audio_signal.shape) == [1, expected_audio_samples]

    # Extract tokens
    audio_tokens = token_model.wav2token(audio_signal.to(device))
    expected_num_tokens = int(expected_frame_rate * audio_duration_sec)
    assert list(audio_tokens.shape) == [1, expected_num_tokens]

    print(f"ConformerUMM_TTS_TOPE_LFR {expected_frame_rate}hz succesfully loaded and tested: {ckpt_path}")


def test_model(model_key, expected_frame_rate):
    ckpt_path = MODELS_DICT[model_key]
    unique_cache_dir = f"./.{model_key}"
    run_model(ckpt_path, unique_cache_dir, expected_frame_rate)


if __name__ == "__main__":
    """The longest amount of time is spent downloading the checkpoint. By default, just test the model you are interested in."""
    test_model("conformerumm_tts_rope_lfr_15hz", 15)
    #test_model("conformerumm_tts_rope_lfr_15hz_larger_codebook", 15)
    #test_model("conformerumm_tts_rope_lfr_20hz", 20)
    #test_model("conformerumm_tts_rope_lfr_20hz_larger_codebook", 20)
