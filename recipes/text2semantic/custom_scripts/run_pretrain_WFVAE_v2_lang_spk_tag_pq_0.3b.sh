'''
### train
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource;
bash launch.sh fit \
        --config recipes/text2semantic/conf/llama/vae_llama_ctiga_wds_lang_spk_tag.yaml \
        --run_opts.data_id 254 \
        --run_opts.hdfs_path hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_merge20231008/text2semantic/pretrain_WFVAE_v2_labv3_punc_data_id254_bt14000_16A100_accu5_byteT5_scr0.15_freezeTrue_langFalse_serTrue_serlossTrue_test \
        --run_opts.log_dir ./logs \
        --run_opts.log_name pretrain_WFVAE_v2_labv3_punc \
        --run_opts.version data_id254_bt14000_16A100_accu5_byteT5_scr0.15_freezeTrue_langFalse_test \
        --run_opts.batch_total_tokens 14000 \
        --run_opts.text_encoder_type byte-T5-base \
        --run_opts.text_encoder_path resource/models/byte-T5-base \
        --run_opts.freeze_text_encoder True \
        --run_opts.strategy ddp_find_unused_parameters_true \
        --run_opts.wds2tag recipes/text2semantic/datasets/urls_lst/wds2tag.data_id254.json \
        --run_opts.spk_cfg_rate 0.0 \
        --run_opts.use_lang_id False \
        --run_opts.use_ser_tag True \
        --run_opts.use_ser_tag_loss True

### infer
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource;
bash launch.sh predict \
        -c recipes/text2semantic/conf/llama/inference_wvae_icl_lang_spk_tag.yaml \
        --run_opts.meta_lst /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/temp_en/meta.lst.conversation_20 \
        --run_opts.ckpt_path hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/huangzhiying.92/exp/samantha_bigtts_merge20231008/text2semantic/pretrain_WFVAE_v2_labv3_punc_data_id192_bt14000_16A100_accu5_byteT5_scr0.15_freezeTrue_langFalse/checkpoints/last.ckpt \
        --run_opts.output_dir /mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_merge20231008/text2semantic/pretrain_WFVAE_v2_labv3_punc_data_id192_bt14000_16A100_accu5_byteT5_scr0.15_freezeTrue_langFalse/infer/test \
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
data_id=682
spk_cfg_rate=0.15
use_sp=False
train_ckpt_path=

. /opt/tiger/samantha/scripts/parse_options.sh

if [ $mode == "debug" ]; then
    work_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_ref_enc
fi

if [ $mode == "train" ]; then
    work_dir=/opt/tiger/samantha
fi

if [ ${stage} -eq 1 ];then
    log_name=pretrain_WFVAE_v2_labv3_punc_0.3b_debug
    version=data_id${data_id}_bt${batch_total_tokens}_16A100_accu5_scr${spk_cfg_rate}_sp${use_sp}
    hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/${log_name}_${version}

    cd $work_dir

    if [[ ! -d ./resource ]]; then
        hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource
    fi

    export LD_LIBRARY_PATH=/opt/tiger/jdk/jdk1.8/jre/lib/amd64/server:/opt/tiger/yarn_deploy/hadoop_current/lib/native:/opt/tiger/yarn_deploy/hadoop_current/lib/native/ufs:/opt/tiger/yarn_deploy/hadoop/lib/native:/opt/tiger/yarn_deploy/hadoop_current/lib/native:/opt/tiger/yarn_deploy/hadoop_current/lzo/lib:/usr/local/cuda/lib64::/usr/local/cuda/compat;

    train_ckpt_path=${train_ckpt_path:-$hdfs_path/checkpoints/last.ckpt}
    base_config="--config recipes/text2semantic/conf/llama/vae_llama_ctiga_parquet_lang_spk_tag_ser.yaml \
                --run_opts.data_id ${data_id} \
                --run_opts.hdfs_path $hdfs_path \
                --run_opts.log_dir ./logs \
                --run_opts.log_name ${log_name} \
                --run_opts.version ${version} \
                --run_opts.batch_total_tokens ${batch_total_tokens} \
                --run_opts.text_encoder_type byte-T5-base \
                --run_opts.text_encoder_path resource/models/byte-T5-base \
                --run_opts.strategy ddp_find_unused_parameters_true \
                --run_opts.spk2tag recipes/text2semantic/datasets/dict/spk2tag.parquet.json \
                --run_opts.spk2id recipes/text2semantic/datasets/dict/spk2id.parquet.json \
                --run_opts.spk_cfg_rate ${spk_cfg_rate} \
                --run_opts.use_sp ${use_sp} \
                --llama_config.dim 1024 \
                --llama_config.n_layers 24 \
                --llama_config.n_heads 16 \
                --llama_config.cross_attention_n_heads 16"

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
    hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_ref_enc/text2semantic/pretrain_WFVAE_v2_labv3_punc_0.3b_data_id539_bt25000_16A100_accu5_scr0.15_spFalse
    local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_ref_enc/text2semantic/pretrain_WFVAE_v2_labv3_punc_0.3b_data_id539_bt25000_16A100_accu5_scr0.15_spFalse
    ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '85000'`
    for ar_ckpt_path in $ar_ckpt_paths;do
        step=`echo $ar_ckpt_path | awk -F"/" '{print $NF}' | cut -d "-" -f 2 | cut -d "=" -f 2`
        for infer_spk_name in tts_Lmand_Sinhouse-multistyle_P1/xudong_yqc_1105 tts_Lmand_Sinhouse-multistyle_P1/M392_conv_1107_del_er tts_Lmand_Sinhouse-multistyle_P1/sophie_conv_1020;do
            infer_spk_basename=`echo $infer_spk_name | cut -d "/" -f 2`
            tag_id=3
            for pair in flow_tts_zh-listen_all;do
                cd $work_dir
                if [[ ! -d ./resource ]]; then
                    hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource
                fi

                testspk=`echo $pair | cut -d "-" -f 1`
                testset=`echo $pair | cut -d "-" -f 2`

                lang='en'
                if [ $pair == "flow_tts_zh-wer_rand20" -o $pair == "flow_tts_zh-yuqici" -o $pair == "flow_tts_zh-temp_demo20231104_v7" -o $pair == "flow_tts_zh-deploy_test" -o $pair == "flow_tts_zh-erhuayin" ]; then
                    lang='zh'
                fi

                cal_asv='FALSE'

                metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}
                out_wav_dir=$local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}/wav

                bash launch.sh predict \
                    -c recipes/text2semantic/conf/llama/inference_wvae_icl_lang_spk_tag.yaml \
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
                    --run_opts.spk2id recipes/text2semantic/datasets/dict/spk2id.parquet.json \
                    --run_opts.use_sp False

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
                python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/analyse/analyse_wer_listentest_single.py \
                                                                analyse \
                                                                ${out_wav_dir}/../wav_res_ref_text.wer \
                                                                ${out_wav_dir}/.. \
                                                                $lang

                if [ $pair == "flow_tts_zh-deploy_test" ]; then
                    python3 recipes/text2semantic/utils/merge_wav/merge_wav.py ${out_wav_dir} ${out_wav_dir}_merge
                fi
            done
        done
    done
fi


# infer seed1996
if [ ${stage} -eq 3 ];then
    hdfs_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/huangzhiying.92/exp/samantha_bigtts_merge20231008/text2semantic/pretrain_WFVAE_v2_labv3_punc_data_id192_bt14000_16A100_accu5_byteT5_scr0.15_freezeTrue_langFalse
    local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_merge20231008/text2semantic/pretrain_WFVAE_v2_labv3_punc_data_id192_bt14000_16A100_accu5_byteT5_scr0.15_freezeTrue_langFalse
    ar_ckpt_path=$hdfs_path/checkpoints/epoch=00-step=65000-kl_loss=0.66.ckpt
    step=`echo $ar_ckpt_path | awk -F"/" '{print $NF}' | cut -d "-" -f 2 | cut -d "=" -f 2`
    tag_id=2
    for pair in temp_en-test20231013;do
        cd $work_dir
        if [[ ! -d ./resource ]]; then
            hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource
        fi

        testspk=`echo $pair | cut -d "-" -f 1`
        testset=`echo $pair | cut -d "-" -f 2`

        lang='en'
        if [ $testspk == "inner_testset_zh" -o $testspk == "out_testset_zh" -o $testspk == "temp" -o $testspk == "cross_lingual_en_zh" -o $testspk == "inner_testset_en2zh" -o $testspk == "out_testset_en2zh" -o $testspk == "201_zh2en" ]; then
            lang='zh'
        fi

        cal_asv='FALSE'
        # if [ $testspk == "temp" ]; then
        #     cal_asv='False'
        # fi

        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}
        out_wav_dir=$local_path/infer/${pair}/step${step}_tag_id${tag_id}/wav
        # [ -d $out_wav_dir ] && continue

        bash launch.sh predict \
            -c recipes/text2semantic/conf/llama/inference_wvae_icl_lang_spk_tag.yaml \
            --run_opts.meta_lst $metalst \
            --run_opts.ckpt_path $ar_ckpt_path \
            --run_opts.output_dir $out_wav_dir \
            --run_opts.wvae_encoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae/wavevae_encoder_%d.pt \
            --run_opts.wvae_decoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae/wavevae_decoder_%d.pt \
            --run_opts.ar_model_name 'VAET2SLangSpkModule' \
            --run_opts.infer_spk_name 'default' \
            --run_opts.tag_id $tag_id \
            --run_opts.tokenizer_type byte-T5-base \
            --run_opts.bpe_dir resource/models/byte-T5-base \
            --run_opts.use_prompt True

        # asr_type=internal
        # metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}
        # python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/utils/get_wav_res_ref_text.py $metalst ${out_wav_dir} ${out_wav_dir}/../wav_res_ref_text
        # bigtts_eval_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval
        # bash /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/run-eval_parse.sh \
        #                                                 --bigtts_eval_dir $bigtts_eval_dir \
        #                                                 --asr_type $asr_type \
        #                                                 --infile ${out_wav_dir}/../wav_res_ref_text \
        #                                                 --outdir ${out_wav_dir}/../ \
        #                                                 --lang $lang \
        #                                                 --cal_asv $cal_asv
        # python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/analyse/analyse_wer_listentest_single.py \
        #                                                 analyse \
        #                                                 ${out_wav_dir}/../wav_res_ref_text.wer \
        #                                                 ${out_wav_dir}/.. \
        #                                                 $lang
    done
fi
