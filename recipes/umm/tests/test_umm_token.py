import os
import random

import torch

import scripts.data_processing.audio.umm_token as umm_token
from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.requires.model_initializer import ensure_hdfs_ckpt_is_local

"""
24APR2024 @hanoihantrakul
These tests allow me to quickly check if my newly trained checkpoints will be
compatible with the expected signatures required for the automated BigSpeech
pipeline for launching AR Training for CN Lyrics2Song and Diffusion training. 
"""

MODELS_DICT = {
    # 24APR2024 @hanoihantrakul ConvUMM trained on 2250mixedZHEN+2375Speech data
    "ConvUMM_2250_2375": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conv_mixedZHEN/umm_stage3_conv1D_v2_2250mixedZHEN_2375Speech_direct_stage3_optim_mem_EMAVQ32768x32/checkpoints/step=0310000.ckpt",
    # 24APR2024 @hanoihantrakul ConformerUMM trained on 2250mixedZHEN+2375Speech data (I had to pick up training from an old checkpoint. In reality this model has been trained for 270K steps)
    "ConformerUMM_2250_2375": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conformer_2255mixedZHEN_2375Speech/umm_stage3_2255mixedZHEN_2375Speech_optim_mem_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0150000.ckpt",
    # 24APR2024 @hanoihantrakul
    "ConvUMM_440_713_715": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conv_mix440_713_715/umm_stage3_conv1D_v2_mix440_713_715_direct_stage3_EMAVQ32768x32/checkpoints/step=0380000.ckpt",
}

if __name__ == "__main__":
    """
    Check the model is indeed a Stage3 pl_module and compatible with umm_token.load_model and umm_token.process_batch
    """
    MODEL_KEY = "ConvUMM_440_713_715"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AUDIO_NUM_SECS = 2
    TOKENIZER_FRAME_RATE = 25

    # `umm_token.process_batch()` returns a generator. So I also pass it a generator as an argument.
    def batch_generator(num_repeats=5):
        for _ in range(num_repeats):
            # Generate random 2 second sin test signal
            audio_signal = pitch_utils.gen_test_sin(
                freq=random.randrange(220, 880),
                amp=0.9,
                dur_sec=15,
                sr=24000,
                add_batch_dim=True,
            ).to(DEVICE)
            assert list(audio_signal.shape) == [1, AUDIO_NUM_SECS * 24000]
            yield audio_signal

    audio_batch = batch_generator()

    # Download and load checkpoint
    hpath = MODELS_DICT[MODEL_KEY]
    cache_dir = "./." + MODEL_KEY
    os.makedirs(cache_dir, exist_ok=True)
    local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
    token_model = umm_token.load_model(DEVICE, local_path)

    # umm_token.process_batch returns a generator that has to be looped over
    output_generator = umm_token.process_batch(token_model, audio_batch, DEVICE)
    for output in output_generator:
        # Under the hood, the token_model.wav2token() operation is only called when the generator is called and yields an output
        assert output.shape == (
            TOKENIZER_FRAME_RATE * AUDIO_NUM_SECS,
        )  # We expect 2 seconds of 25Hz audio tokens

    print(f"Succesfully checked Stage3 compatability for tokenizer: {local_path}")
