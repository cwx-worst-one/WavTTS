'''
### train
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource;
bash launch.sh fit \
        --config recipes/text2semantic/conf/llama/vae_llama_ctiga_wds_lang_spk_tag.yaml \
        --run_opts.data_id 234 \
        --run_opts.hdfs_path hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_merge20231008/text2semantic/sft_WFVAE_v2_labv3_punc_data_id234_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse_test \
        --run_opts.log_dir ./logs \
        --run_opts.log_name sft_WFVAE_v2_labv3_punc \
        --run_opts.version data_id234_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse_test \
        --run_opts.batch_total_tokens 14000 \
        --run_opts.text_encoder_type byte-T5-base \
        --run_opts.text_encoder_path resource/models/byte-T5-base \
        --run_opts.freeze_text_encoder True \
        --run_opts.strategy ddp_find_unused_parameters_true \
        --run_opts.wds2tag recipes/text2semantic/datasets/urls_lst/wds2tag.data_id234.json \
        --run_opts.spk_cfg_rate 0.0 \
        --run_opts.use_lang_id False \
        --ckpt_path hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/huangzhiying.92/exp/samantha_bigtts_merge20231008/text2semantic/pretrain_WFVAE_v2_labv3_punc_data_id192_bt14000_16A100_accu5_byteT5_scr0.15_freezeTrue_langFalse/checkpoints/last.ckpt

### infer
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource;
bash launch.sh predict \
        -c recipes/text2semantic/conf/llama/inference_wvae_icl_lang_spk_tag.yaml \
        --run_opts.meta_lst /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/temp_en/meta.lst.conversation_20 \
        --run_opts.ckpt_path hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_merge20231008/text2semantic/sft_WFVAE_v2_labv3_punc_data_id234_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse_20231016/checkpoints/last.ckpt \
        --run_opts.output_dir /mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_merge20231008/text2semantic/sft_WFVAE_v2_labv3_punc_data_id234_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse_20231016/infer/test \
        --run_opts.wvae_encoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_1.0/wavevae_encoder_%d.pt \
        --run_opts.wvae_decoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_1.0/wavevae_decoder_%d.pt \
        --run_opts.ar_model_name "VAET2SLangSpkModule" \
        --run_opts.infer_spk_name duibiao/sarah_conversation \
        --run_opts.tag_id 3 \
        --run_opts.tokenizer_type byte-T5-base \
        --run_opts.bpe_dir resource/models/byte-T5-base
'''

set -x

stage=1
batch_total_tokens=14000
mode=debug  # train or debug
data_id=553
spk_cfg_rate=0.0
freeze_text_encoder=True
use_lang_id=False
accumulate_grad_batches=5
train_ckpt_path=

. /opt/tiger/samantha/scripts/parse_options.sh

if [ $mode == "debug" ]; then
    work_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_ref_enc
fi

if [ $mode == "train" ]; then
    work_dir=/opt/tiger/samantha
fi

if [ ${stage} -eq 1 ];then
    log_name=sft_WFVAE_v2_labv3_punc
    version=data_id${data_id}_bt${batch_total_tokens}_16A100_accu${accumulate_grad_batches}_byteT5_scr${spk_cfg_rate}_freeze${freeze_text_encoder}_lang${use_lang_id}
    hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/${log_name}_${version}

    cd $work_dir

    if [[ ! -d ./resource ]]; then
        hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource
    fi

    train_ckpt_path=${train_ckpt_path:-$hdfs_path/checkpoints/last.ckpt}
    base_config="--config recipes/text2semantic/conf/llama/vae_llama_ctiga_wds_lang_spk_tag.yaml \
                --run_opts.data_id ${data_id} \
                --run_opts.hdfs_path $hdfs_path \
                --run_opts.log_dir ./logs \
                --run_opts.log_name ${log_name} \
                --run_opts.version ${version} \
                --run_opts.batch_total_tokens ${batch_total_tokens} \
                --run_opts.text_encoder_type byte-T5-base \
                --run_opts.text_encoder_path resource/models/byte-T5-base \
                --run_opts.freeze_text_encoder ${freeze_text_encoder} \
                --run_opts.strategy ddp_find_unused_parameters_true \
                --run_opts.wds2tag recipes/text2semantic/datasets/urls_lst/wds2tag.data_id${data_id}.json \
                --run_opts.spk_cfg_rate ${spk_cfg_rate} \
                --run_opts.use_lang_id ${use_lang_id} \
                --run_opts.accumulate_grad_batches ${accumulate_grad_batches}"

    if hdfs dfs -test -e ${train_ckpt_path}; then
        bash launch.sh fit \
            $base_config \
            --ckpt_path $train_ckpt_path
    else
        bash launch.sh fit \
            $base_config
    fi
fi


# infer seed1996
if [ ${stage} -eq 2 ];then
    # # sophie, sinong, maomao
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_wvae_sft_tobe_merge/text2semantic/sft_WFVAE_v2_labv3_punc_data_id307_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id307_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '118500'`

    # # M392
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id369_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id369_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '100000'`

    # # taozi, xudong
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id422_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id422_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '95000'`

    # # taozi_20231112, xudong_20231112, M392_20231112, sophie_20231112, sinong
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id439_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id439_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '95000'`
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '170000'`
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '125000'`
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '100000|105000|110000|115000|120000|125000|130000|135000|140000|145000|150000|155000|160000|165000|170000|90000|95000'`

    # # F274_conv, charlie_conv
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id454_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id454_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '100000|105000|110000|115000|120000|125000|130000|90000|95000'`

    # # xudong_1011_1028_1106_p100
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id515_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id515_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '90000|95000|100000'`

    # # xudong_1011_1028_1106_p110
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id538_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id538_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '95000'`

    # # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id553_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id553_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '95000'`


    hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id590_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id590_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '95000'`

    # for retry in {0..9};do
    for ar_ckpt_path in $ar_ckpt_paths;do
        step=`echo $ar_ckpt_path | awk -F"/" '{print $NF}' | cut -d "-" -f 2 | cut -d "=" -f 2`
        # for infer_spk_name in duibiao/maomao_conversation;do
        # for infer_spk_name in duibiao/taozi_conversation duibiao/sophie_conv_1019 duibiao/M392_conv duibiao/xudong_yqc_1105;do
        # for infer_spk_name in duibiao/taozi_conversation;do
        # for infer_spk_name in duibiao/sinong_conv_1103_update;do
        # for infer_spk_name in duibiao/sophie_conv_1019;do
        # for infer_spk_name in duibiao/xudong_yqc_1105;do
        # for infer_spk_name in duibiao/xudong_1011_1028_1106_p100;do
        # for infer_spk_name in duibiao/sophie_conv_1019;do
        # for infer_spk_name in duibiao/sophie_conv_1014_1119_update duibiao/xudong_conv_1119_update duibiao/M392_conv_1119_update duibiao/M525_conv_1119 duibiao/M223_conv_1119;do
        # for infer_spk_name in duibiao/M392_conv;do
        for infer_spk_name in duibiao/M525_conv_1119;do
            infer_spk_basename=`echo $infer_spk_name | cut -d "/" -f 2`
            tag_id=3
            # for pair in flow_tts_zh-listen_all_new_zh_select;do
            # for pair in flow_tts_zh-listen_all_new_en_select flow_tts_zh-listen_all_new_codeswitch_select flow_tts_zh-listen_all_new_zh_select flow_tts_zh-listen_all_new_others_123;do
            # for pair in flow_tts_zh-yuqici;do
            # for pair in flow_tts_zh-temp_demo20231104_v7;do
            # for pair in flow_tts_zh-temp_for_naihan;do
            # for pair in flow_tts_zh-temp_sinong_vol;do
            # for pair in flow_tts_zh-temp_sinong_en;do
            # for pair in flow_tts_zh-yuqici_rand50;do
            # for pair in flow_tts_zh-listen_all_new_others_123_v2;do
            for pair in flow_tts_zh-listen_all_new_others_123_v2 flow_tts_zh-listen_all_new_en_select flow_tts_zh-listen_all_new_codeswitch_select_v2 flow_tts_zh-listen_all_new_zh_select_v2;do
            # for pair in flow_tts_zh-test;do
            # for pair in flow_tts_zh-listen_all_new_zh_select_v2;do
                cd $work_dir
                if [[ ! -d ./resource ]]; then
                    hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource
                fi

                testspk=`echo $pair | cut -d "-" -f 1`
                testset=`echo $pair | cut -d "-" -f 2`

                lang='en'
                if [ $pair == "flow_tts_zh-wer_rand20" -o $pair == "flow_tts_zh-yuqici" -o $pair == "flow_tts_zh-temp_demo20231104_v7" -o $pair == "flow_tts_zh-deploy_test" -o $pair == "flow_tts_zh-erhuayin" -o $pair == "flow_tts_zh-yuqici_haha" -o $pair == "flow_tts_zh-listen_all_new_zh_select" -o $pair == "flow_tts_zh-listen_all_new_codeswitch_select" -o $pair == "flow_tts_zh-listen_all_new_others_123_v2" -o $pair == "flow_tts_zh-yuqici_rand50" ]; then
                    lang='zh'
                fi

                cal_asv='FALSE'

                metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}
                out_wav_dir=$local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse_online_shortform/wav

                bash launch.sh predict \
                    -c recipes/text2semantic/conf/llama/inference_wvae_icl_lang_spk_tag_online.yaml \
                    --run_opts.meta_lst $metalst \
                    --run_opts.ckpt_path $ar_ckpt_path \
                    --run_opts.output_dir $out_wav_dir \
                    --run_opts.wvae_encoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_1.0/wavevae_encoder_%d.pt \
                    --run_opts.wvae_decoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_1.0/wavevae_decoder_%d.pt \
                    --run_opts.ar_model_name 'VAET2SLangSpkModule' \
                    --run_opts.infer_spk_name $infer_spk_name \
                    --run_opts.tag_id $tag_id \
                    --run_opts.tokenizer_type byte-T5-base \
                    --run_opts.bpe_dir resource/models/byte-T5-base \
                    --run_opts.use_sp False \
                    --run_opts.max_paragraph_phoneme_size_zh 240 \
                    --run_opts.max_paragraph_phoneme_size_en 480 \
                    --run_opts.sil_interval 0.3
                    # --run_opts.infer_tacolab_dir /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/flow_tts_zh/test

                python3 recipes/text2semantic/utils/analysis_pitch.py $out_wav_dir $out_wav_dir/../pitch.txt &> $out_wav_dir/../pitch.log
                cp $metalst $out_wav_dir/../
                asr_type=internal
                metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}
                python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/utils/get_wav_res_ref_text.py $metalst ${out_wav_dir} ${out_wav_dir}/../wav_res_ref_text
                bigtts_eval_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval
                bash /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/run-eval_parse.sh \
                                                                --bigtts_eval_dir $bigtts_eval_dir \
                                                                --asr_type $asr_type \
                                                                --infile ${out_wav_dir}/../wav_res_ref_text \
                                                                --outdir ${out_wav_dir}/../ \
                                                                --lang $lang \
                                                                --cal_asv $cal_asv
                [ -d ${out_wav_dir}/../analyse ] && rm -rf ${out_wav_dir}/../analyse
                python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/analyse/analyse_wer_listentest_single.py \
                                                                analyse \
                                                                ${out_wav_dir}/../wav_res_ref_text.wer \
                                                                ${out_wav_dir}/.. \
                                                                $lang
            done
        done
    done
    # done
fi


# infer seed1996
if [ ${stage} -eq 3 ];then
    # # sophie, sinong, maomao
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_wvae_sft_tobe_merge/text2semantic/sft_WFVAE_v2_labv3_punc_data_id307_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id307_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '118500'`

    # # M392
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id369_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id369_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '100000'`

    # # taozi, xudong
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id422_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id422_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '95000'`

    # # taozi_20231112, xudong_20231112, M392_20231112, sophie_20231112, sinong
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id439_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id439_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '95000'`
    # # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '170000'`

    # # F274_conv, charlie_conv
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id454_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id454_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '100000|105000|110000|115000|120000|125000|130000|90000|95000'`

    # # xudong_1011_1028_1106_p100
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id515_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id515_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '90000|95000|100000'`

    # # xudong_1011_1028_1106_p110, taozi_new
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id538_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id538_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '95000'`

    # xudong_new
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id553_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id553_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '95000'`

    # sophie_new
    hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id590_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id590_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '95000'`

    # # sinong_new
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/liuxudong/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id589_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/sft_WFVAE_v2_labv3_punc_data_id589_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '95000'`

    # for retry in {0..9};do
    for ar_ckpt_path in $ar_ckpt_paths;do
        step=`echo $ar_ckpt_path | awk -F"/" '{print $NF}' | cut -d "-" -f 2 | cut -d "=" -f 2`
        # for infer_spk_name in duibiao/maomao_conversation;do
        # for infer_spk_name in duibiao/taozi_conversation duibiao/sophie_conv_1019 duibiao/M392_conv duibiao/xudong_yqc_1105;do
        # for infer_spk_name in duibiao/taozi_conversation;do
        # for infer_spk_name in duibiao/sinong_conv_1103_update;do
        for infer_spk_name in duibiao/sophie_conv_1019;do
        # for infer_spk_name in duibiao/xudong_yqc_1105;do
        # for infer_spk_name in duibiao/xudong_1011_1028_1106_p100;do
        # for infer_spk_name in duibiao/sophie_conv_1019;do
        # for infer_spk_name in duibiao/sophie_conv_1014_1119_update duibiao/xudong_conv_1119_update duibiao/M392_conv_1119_update duibiao/M525_conv_1119 duibiao/M223_conv_1119;do
        # for infer_spk_name in duibiao/M392_conv;do
            infer_spk_basename=`echo $infer_spk_name | cut -d "/" -f 2`
            tag_id=3
            # for pair in flow_tts_zh-listen_all_new_zh_select;do
            # for pair in flow_tts_zh-listen_all_new_en_select flow_tts_zh-listen_all_new_codeswitch_select flow_tts_zh-listen_all_new_zh_select flow_tts_zh-listen_all_new_others_123;do
            # for pair in flow_tts_zh-yuqici;do
            # for pair in flow_tts_zh-temp_demo20231104_v7;do
            # for pair in flow_tts_zh-temp_for_naihan;do
            # for pair in flow_tts_zh-temp_sinong_vol;do
            # for pair in flow_tts_zh-temp_sinong_en;do
            # for pair in flow_tts_zh-yuqici_rand50;do
            # for pair in flow_tts_zh-listen_all_new_others_123_v2;do
            # for pair in flow_tts_zh-listen_all_new_en_select flow_tts_zh-listen_all_new_codeswitch_select_v2 flow_tts_zh-listen_all_new_zh_select_v2;do
            # for pair in flow_tts_zh-test;do
            for pair in flow_tts_zh-temp_for_sophie;do
                cd $work_dir
                if [[ ! -d ./resource ]]; then
                    hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource
                fi

                testspk=`echo $pair | cut -d "-" -f 1`
                testset=`echo $pair | cut -d "-" -f 2`

                lang='en'
                if [ $pair == "flow_tts_zh-wer_rand20" -o $pair == "flow_tts_zh-yuqici" -o $pair == "flow_tts_zh-temp_demo20231104_v7" -o $pair == "flow_tts_zh-deploy_test" -o $pair == "flow_tts_zh-erhuayin" -o $pair == "flow_tts_zh-yuqici_haha" -o $pair == "flow_tts_zh-listen_all_new_zh_select" -o $pair == "flow_tts_zh-listen_all_new_codeswitch_select" -o $pair == "flow_tts_zh-listen_all_new_others_123_v2" -o $pair == "flow_tts_zh-yuqici_rand50" -o $pair == "flow_tts_zh-listen_all_new_zh_select_v2" ]; then
                    lang='zh'
                fi

                cal_asv='FALSE'

                metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}
                out_wav_dir=$local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse_online_shortform/wav
                # out_wav_dir=$local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse_online_shortform_streaming/wav

                bash launch.sh predict \
                    -c recipes/text2semantic/conf/llama/inference_wvae_icl_lang_spk_tag_online.yaml \
                    --run_opts.meta_lst $metalst \
                    --run_opts.ckpt_path $ar_ckpt_path \
                    --run_opts.output_dir $out_wav_dir \
                    --run_opts.wvae_encoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_1.0/wavevae_encoder_%d.pt \
                    --run_opts.wvae_decoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_1.0/wavevae_decoder_%d.pt \
                    --run_opts.ar_model_name 'VAET2SLangSpkModule' \
                    --run_opts.infer_spk_name $infer_spk_name \
                    --run_opts.tag_id $tag_id \
                    --run_opts.tokenizer_type byte-T5-base \
                    --run_opts.bpe_dir resource/models/byte-T5-base \
                    --run_opts.use_sp False \
                    --run_opts.max_paragraph_phoneme_size_zh 240 \
                    --run_opts.max_paragraph_phoneme_size_en 480 \
                    --run_opts.sil_interval 0.3

                python3 recipes/text2semantic/utils/analysis_pitch.py $out_wav_dir $out_wav_dir/../pitch.txt &> $out_wav_dir/../pitch.log
                cp $metalst $out_wav_dir/../
                asr_type=internal
                metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}
                python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/utils/get_wav_res_ref_text.py $metalst ${out_wav_dir} ${out_wav_dir}/../wav_res_ref_text
                bigtts_eval_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval
                bash /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/run-eval_parse.sh \
                                                                --bigtts_eval_dir $bigtts_eval_dir \
                                                                --asr_type $asr_type \
                                                                --infile ${out_wav_dir}/../wav_res_ref_text \
                                                                --outdir ${out_wav_dir}/../ \
                                                                --lang $lang \
                                                                --cal_asv $cal_asv
                [ -d ${out_wav_dir}/../analyse ] && rm -rf ${out_wav_dir}/../analyse
                python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/analyse/analyse_wer_listentest_single.py \
                                                                analyse \
                                                                ${out_wav_dir}/../wav_res_ref_text.wer \
                                                                ${out_wav_dir}/.. \
                                                                $lang
            done
        done
    done
    # done
fi