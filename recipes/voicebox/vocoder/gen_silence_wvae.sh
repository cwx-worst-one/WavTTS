wvae_encoder_path=hdfs://haruna/home/byte_speech_sv/user/liuzhengxi/WaveVAE3.0/torchscript/wavevae_encoder_z1_%d.pt
wvae_decoder_path=hdfs://haruna/home/byte_speech_sv/user/liuzhengxi/WaveVAE3.0/torchscript/wavevae_decoder_z1_%d.pt
wvae_cache_dir=.module_cache/wvae/v3_40hz
version=3.1
out_dir=recipes/voicebox/vocoder/silence_wvae/v3_40hz/

python3 recipes/voicebox/vocoder/gen_silence_wvae.py \
    --wvae_encoder_path $wvae_encoder_path \
    --wvae_decoder_path $wvae_decoder_path \
    --wvae_cache_dir $wvae_cache_dir \
    --version $version \
    --device "cuda:0" \
    --out_dir $out_dir