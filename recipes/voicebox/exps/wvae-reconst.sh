mode=$1

# wvae_encoder_path=hdfs://haruna/home/byte_speech_sv/user/liuzhengxi/WaveVAE2_80hz/torchscript/wavevae_encoder_%d.pt
# wvae_decoder_path=hdfs://haruna/home/byte_speech_sv/user/liuzhengxi/WaveVAE2_80hz/torchscript/wavevae_decoder_%d.pt
# cache_dir=.module_cache/wvae/v2_80hz/


# wvae_encoder_path=hdfs://haruna/home/byte_speech_sv/user/liuzhengxi/WaveVAE2_200hz/torchscript/wavevae_encoder_%d.pt
# wvae_decoder_path=hdfs://haruna/home/byte_speech_sv/user/liuzhengxi/WaveVAE2_200hz/torchscript/wavevae_decoder_%d.pt
# cache_dir=.module_cache/wvae/v2_200hz/

# wvae_encoder_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_1.0/wavevae_encoder_%d.pt
# wvae_decoder_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_1.0/wavevae_decoder_%d.pt
# cache_dir=.module_cache/wvae/v2_60hz

wvae_encoder_path=hdfs://haruna/home/byte_speech_sv/user/liuzhengxi/WaveVAE2_40hz/wavevae_encoder_%d.pt
wvae_decoder_path=hdfs://haruna/home/byte_speech_sv/user/liuzhengxi/WaveVAE2_40hz//wavevae_decoder_%d.pt
cache_dir=.module_cache/wvae/v2_40hz

workdir=$PWD
if [[ ${mode} == "wvae" ]];then
    for lang in zh en;
    do
        cd $workdir

        meta_lst=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_2.0/${lang}/reconstruct_meta.lst
        # out_dir=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_2.0/${lang}/wvae-2.0-40hz-xin-reconst/
        out_dir=/mnt/bn/jcong5/tmp/debug/${lang}/wvae-2.0-40hz-xin-reconst/
        
        # meta_lst=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_focus/${lang}_reconstruct_meta.lst
        # out_dir=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_focus/${lang}-wvae-2.0-200hz-reconst/

        sudo mkdir -p $out_dir
        sudo chmod 777 -R $out_dir

        # 120、480
        bash launch.sh predict \
            -c recipes/voicebox/conf/infer_reconstruction_40hzMel.yaml \
            --run_opts.meta_lst $meta_lst  \
            --run_opts.output_dir $out_dir \
            --run_opts.umm_frame_rate 40 \
            --run_opts.mel_frame_rate 40 \
            --run_opts.infer_type wvae-recons \
            --run_opts.use_wvae_vocoder True \
            --run_opts.lang $lang \
            --bn_config.wvae_encoder_path $wvae_encoder_path \
            --bn_config.wvae_decoder_path $wvae_decoder_path \
            --bn_config.wvae_version 2.1 \
            --bn_config.wvae_cache_dir $cache_dir \
            --bn_config.wvae_encoder_hop_size 300 \
            --bn_config.wvae_encoder_win_size 1200

        bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang
    done
fi


if [[ ${mode} == "vocoder" ]];then
    vocoder_ckpt_path=/mnt/bn/jcong5/workspace5/samantha-speech-diffusion/.module_cache/ihifigan/500k.pt
    for lang in zh en;
    do
        cd $workdir
        indir=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_2.0/${lang}/wavs/
        meta_lst=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_2.0/${lang}/reconstruct_meta.lst
        out_dir=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_2.0/${lang}/ihifigan-reconst/
        # meta_lst=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_focus/${lang}_reconstruct_meta.lst
        # out_dir=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_focus/${lang}-wvae-2.0-200hz-reconst/

        sudo mkdir -p $out_dir
        sudo chmod 777 -R $out_dir

        python3 recipes/voicebox/scripts/recons_vocoder_40hz.py\
                --vocoder_ckpt_path $vocoder_ckpt_path \
                --test_wav_dir $indir \
                --out_dir $out_dir \
                --device cuda:0
        bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang
    done
fi