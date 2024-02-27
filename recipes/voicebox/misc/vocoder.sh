
# test convert mel to wav using bigvgan

# cd recipes/voicebox/vocoder/BigVGAN
# python3 inference_e2e.py \
#     --input_mels_dir /mnt/bn/jcong5/tmp/test_bigvgan/ \
#     --output_dir /mnt/bn/jcong5/tmp/test_bigvgan/ \
#     --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/bigvgan_24khz_100band/g_05000000.zip 

# test convert mel to wav using hifigan

# cd recipes/voicebox/vocoder/hifi-gan/
# python3 inference_e2e.py \
#     --input_mels_dir /mnt/bn/jcong5/tmp/test_hifigan/ \
#     --output_dir /mnt/bn/jcong5/tmp/test_hifigan/ \
#     --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 