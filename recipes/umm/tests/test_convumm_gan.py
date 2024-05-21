import torch

from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.requires.model_initializer import init_convumm_gan

"""
@hanoihantrakul 20MAY2024
Basic example demonstrating how to load ConvUMM-GAN model.

Prior to this master branch implementation, I spiked different versions of 
ConvUMM-GAN and called it "DualUMM_inst_only" because it literally was
a DualUMM model with only one instrumental branch. After 20MAY2024
I standardized this into a separate new model and called it ConvUMM-GAN
to disambiguate it from DualUMM.
"""

MODELS_DICT = {
    # @hanoihantrakul: 20MAY2024 Instrumental-only version of ConvUMM-GAN (trained on ID 794-SSTK)
    "convumm_gan_sstk": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/convumm_gan_master/convumm_gan_719M_sstk_None_EMAEntropy32768x32/checkpoints/step=0700000.ckpt",
    # @hanoihantrakul: 20MAY2024 Vocal version of ConvUMM-GAN (trained on ID 2255-MixedZHEN and 2375-SpeechKaraoke)
    "convumm_gan_vocal_music": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/convumm_gan_master/convumm_gan_719M_2255_2375_vocals_bert-base-multilingual-uncased_EMAEntropy32768x32/checkpoints/step=0590000.ckpt",
}


def load_model(ckpt_path, cache_dir):
    """The ConvUMM-GAN is based on the DualUMM lit_module. It makes it easier to integrate with existing AR and diffusion pipelines."""
    print(f"Downloading {ckpt_path}")
    DUMMY_RANK = 0
    token_model = init_convumm_gan(ckpt_path, DUMMY_RANK, cache_dir)[
        "convumm_gan_model"
    ].eval()
    """
    @hanoihantrakul 21MAY2024 I leave this example here just for understanding purposes.
    Since ConvUMM-GAN inherits a lot of code from the DualUMM lit_module, it is possible to do:
    `token_model = init_dualumm(ckpt_path, DUMMY_RANK, cache_dir)["Stage3"].eval()`
    However, this is not recommended and is confusing.
    """
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

    print(f"ConvUMM-GAN succesfully loaded and tested: {ckpt_path}")


def test_convumm_gan_sstk_model():
    """20MAY2024 Mainly for Andrew."""
    model_key = "convumm_gan_sstk"
    ckpt_path = MODELS_DICT[model_key]
    unique_cache_dir = f"./.{model_key}"
    run_model(ckpt_path, unique_cache_dir)


def test_convumm_gan_vocal_music_model():
    """20MAY2024 Mainly for QQ and Shuo."""
    model_key = "convumm_gan_vocal_music"
    ckpt_path = MODELS_DICT[model_key]
    unique_cache_dir = f"./.{model_key}"
    run_model(ckpt_path, unique_cache_dir)


if __name__ == "__main__":
    """The longest amount of time is spent downloading the checkpoint. By default, just test the model you are interested in."""
    test_convumm_gan_sstk_model()
    test_convumm_gan_vocal_music_model()
