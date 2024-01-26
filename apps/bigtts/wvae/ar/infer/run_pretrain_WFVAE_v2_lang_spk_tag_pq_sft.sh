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

stage=2
batch_total_tokens=14000
mode=debug  # train or debug
data_id=683
spk_cfg_rate=0.0
use_sp=False
train_ckpt_path=
n_step_save=5000

. /opt/tiger/samantha/scripts/parse_options.sh

if [ $mode == "debug" ]; then
    work_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_streaming
fi

if [ $mode == "train" ]; then
    work_dir=/opt/tiger/samantha
fi

# work_dir=/home/tiger/samantha

if [ ${stage} -eq 1 ];then
    log_name=sft_WFVAE_v2_labv3_punc
    version=data_id${data_id}_bt${batch_total_tokens}_16A100_accu5_scr${spk_cfg_rate}_sp${use_sp}
    hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_streaming/text2semantic/${log_name}_${version}

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
                --run_opts.n_step_save ${n_step_save}"

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
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_streaming/text2semantic/sft_WFVAE_v2_labv3_punc_data_id821_bt14000_16A100_accu5_scr0.0_spFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_streaming/text2semantic/sft_WFVAE_v2_labv3_punc_data_id821_bt14000_16A100_accu5_scr0.0_spFalse
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangwenjie.chenxi/bigtts/exp/samantha_flow_sft/text2semantic/sft_WFVAE_v2_labv3_punc_data_id780_bt14000_16A100_accu5_scr0.0_spFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_streaming/text2semantic/sft_WFVAE_v2_labv3_punc_data_id780_bt14000_16A100_accu5_scr0.0_spFalse
    # hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_streaming/text2semantic/sft_WFVAE_v2_labv3_punc_data_id683_bt14000_16A100_accu5_scr0.0_spFalse
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_streaming/text2semantic/sft_WFVAE_v2_labv3_punc_data_id683_bt14000_16A100_accu5_scr0.0_spFalse
    hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/xieshuangyi/exp/samantha_bigtts_streaming/text2semantic/migrate_wvae_ar/final_align
    # local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_streaming/text2semantic/sft_WFVAE_v2_labv3_punc_data_id683_bt14000_16A100_accu5_scr0.0_spFalse
    local_path=/home/tiger/infer
    # ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '65000|95000|125000|155000|160000|190000'`
    ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '65000' | grep -v '165000'`
    for ar_ckpt_path in $ar_ckpt_paths;do
        ar_ckpt_path='/home/tiger/epoch=00-step=105000-kl_loss=0.36.ckpt'
        # ar_ckpt_path=/home/tiger/epoch=00-step=135000-kl_loss=0.33.ckpt
        step=`echo $ar_ckpt_path | awk -F"/" '{print $NF}' | cut -d "-" -f 2 | cut -d "=" -f 2`
        # for infer_spk_name in tts_Lmand_Sinhouse-multistyle_P1/taozi_conversation tts_Lmand_Sinhouse-multistyle_P1/xudong_conv_1028 tts_Lmand_Sinhouse-multistyle_P1/M392_conv tts_Lmand_Sinhouse-multistyle_P1/maomao_conv_1024_update;do
        # for infer_spk_name in tts_Lmand_Sinhouse-multistyle_P1/sophie_conv_1020;do
        # for infer_spk_name in tts_Lmand_Sinhouse-multistyle_P1/xudong_conv_1011_jingbiao_v2;do
        # for infer_spk_name in tts_Lmand_Sinhouse-sophie_P1/sophie_conv_1014_1119_update tts_Lmand_Sinhouse-multistyle_P1/sinong_conv_1103_update tts_Lmand_Sinhouse-multistyle_P1/taozi_conv_1108 tts_Lmand_Sinhouse-multistyle_P1/xudong_yqc_1105 tts_Lmand_Sinhouse-M392_P1/M392_conv_1121_update tts_Lmand_Sinhouse-multistyle_P1/maomao_conv_1024_update;do
        # for infer_spk_name in tts_Lmand_Sinhouse-PU06_P1/PU06 tts_Lmand_Sinhouse-M683_P1/M683_wangye_1207 tts_Lmand_Sinhouse-sophie_P1/sophie_conv_1014_1119_update tts_Lmand_Sinhouse-multistyle_P1/sinong_conv_1103_update tts_Lmand_Sinhouse-multistyle_P1/taozi_conv_1108 tts_Lmand_Sinhouse-multistyle_P1/xudong_yqc_1105 tts_Lmand_Sinhouse-M392_P1/M392_conv_1121_update tts_Lmand_Sinhouse-multistyle_P1/maomao_conv_1024_update;do
        # for infer_spk_name in tts_Lmand_Sinhouse-multistyle_P1/taozi_conv_1108 tts_Lmand_Sinhouse-multistyle_P1/xudong_yqc_1105 tts_Lmand_Sinhouse-M392_P1/M392_conv_1121_update;do
        # for infer_spk_name in tts_Lmand_Sinhouse-M392_P1/M392_conv_1121_update;do
        # for infer_spk_name in tts_Lmand_Sinhouse-ahu_P1/ahu_conv;do
        # for infer_spk_name in tts_Lmand_Sinhouse-M267_P1/M267_fengbaobao_1130;do
        # for infer_spk_name in tts_Lmand_Sinhouse-multistyle_P1/xudong_yqc_1105;do
        #for infer_spk_name in tts_Lmand_Sinhouse-sophie_P1/sophie_conv_1014_1119_update tts_Lmand_Sinhouse-multistyle_P1/sinong_conv_1103_update tts_Lmand_Sinhouse-multistyle_P1/taozi_conv_1108 tts_Lmand_Sinhouse-multistyle_P1/xudong_yqc_1105 tts_Lmand_Sinhouse-M392_P1/M392_conv_1121_update;do
        for infer_spk_name in tts_Lmand_Sinhouse-sophie_P1/sophie_conv_1014_1119_update;do
            infer_spk_basename=`echo $infer_spk_name | cut -d "/" -f 2`
            tag_id=3
            # for pair in flow_tts_zh-listen_all_new_zh_select_v2 flow_tts_zh-listen_all_new_en_select flow_tts_zh-listen_all_new_codeswitch_select_v2;do
            for pair in flow_tts_zh-listen_all_new_zh_select_v2;do
            # for pair in flow_tts_zh-listen_all_new_others_123_v2;do
            # for pair in flow_tts_zh-wer;do
                cd $work_dir
                if [[ ! -d ./resource ]]; then
                    hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource
                fi

                testspk=`echo $pair | cut -d "-" -f 1`
                testset=`echo $pair | cut -d "-" -f 2`

                lang='en'
                if [ $pair == "flow_tts_zh-wer_rand20" -o $pair == "flow_tts_zh-yuqici" -o $pair == "flow_tts_zh-temp_demo20231104_v7" -o $pair == "flow_tts_zh-deploy_test" -o $pair == "flow_tts_zh-erhuayin" -o $pair == "flow_tts_zh-yuqici_haha" -o $pair == "flow_tts_zh-listen_all_new_zh_select_v2" -o $pair == "flow_tts_zh-listen_all_new_codeswitch_select_v2" -o $pair == "flow_tts_zh-wer" ]; then
                    lang='zh'
                fi

                cal_asv='FALSE'

                metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}
                out_wav_dir=$local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse_online/wav

                if [ $pair == "flow_tts_zh-listen_all_new_zh_select_v2" ]; then
                    wav_num=`ls $local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse/wav | wc -l`
                    wer_num=`wc -l $local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse/wav_res_ref_text.wer | awk '{print $1}'`
                    asv_num=`wc -l $local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse/wav_res_ref_text.asv | awk '{print $1}'`
                    if [[ $wav_num == 92 && $wer_num == 95 && $asv_num == 87 ]];then
                        continue
                    fi
                elif [ $pair == "flow_tts_zh-listen_all_new_codeswitch_select_v2" ]; then
                    wav_num=`ls $local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse/wav | wc -l`
                    wer_num=`wc -l $local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse/wav_res_ref_text.wer | awk '{print $1}'`
                    asv_num=`wc -l $local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse/wav_res_ref_text.asv | awk '{print $1}'`
                    if [[ $wav_num == 20 && $wer_num == 23 && $asv_num == 16 ]];then
                        continue
                    fi
                elif [ $pair == "flow_tts_zh-listen_all_new_en_select" ]; then
                    wav_num=`ls $local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse/wav | wc -l`
                    wer_num=`wc -l $local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse/wav_res_ref_text.wer | awk '{print $1}'`
                    asv_num=`wc -l $local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse/wav_res_ref_text.asv | awk '{print $1}'`
                    if [[ $wav_num == 20 && $wer_num == 23 && $asv_num == 16 ]];then
                        continue
                    fi
                fi

                bash launch.sh predict \
                    -c apps/bigtts/wvae/ar/conf/inference_wvae_icl_lang_spk_tag_online_byt5.yaml \
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
                    --run_opts.spk2id apps/bigtts/wvae/ar/resources/spk2id.parquet.json \
                    --run_opts.use_sp False \
                    --run_opts.max_paragraph_phoneme_size_zh 240 \
                    --run_opts.max_paragraph_phoneme_size_en 480 \
                    --run_opts.sil_interval 0.3 \
                    --run_opts.save_tacolab True

                python3 apps/bigtts/wvae/ar/infer/analysis_pitch.py $out_wav_dir $out_wav_dir/../pitch.txt &> $out_wav_dir/../pitch.log
                cp $metalst $out_wav_dir/../
                asr_type=internal
                metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}

                prompt_wav=""
                prompt_txt=""
                if [ $infer_spk_name == "tts_Lmand_Sinhouse-PU06_P1/PU06" ];then
                    prompt_wav=/mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/duibiao/PU06/wav/10000013.wav
                    prompt_txt="我听说过你在蒙德的事迹，所以在刚才的仪式上，稍微关注了你一下。"
                elif [ $infer_spk_name == "tts_Lmand_Sinhouse-M267_P1/M267_fengbaobao_1130" ];then
                    prompt_wav=/mnt/bn/huangzhiying-nas-bigtts-data/lizhonghao/data/bigtts/duibiao/M267_fengbaobao_1130/wav/10000012.wav
                    prompt_txt="如果你不相信我，可以随时联系当地哩警方查证。"
                elif [ $infer_spk_name == "tts_Lmand_Sinhouse-M683_P1/M683_wangye_1207" ];then
                    prompt_wav=/mnt/bn/huangzhiying-nas-bigtts-data/lizhonghao/data/bigtts/duibiao/M683_wangye_1207/wav/30000002.wav
                    prompt_txt="给你算过啦，此时此地这一局，蛇天矫诸事不利，宜休息啊，唉……"
                elif [ $infer_spk_name == "tts_Lmand_Sinhouse-sophie_P1/sophie_conv_1014_1119_update" ];then
                    prompt_wav=/mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/duibiao/sophie_conv_1014_1119_update/wav/1127_2.wav
                    prompt_txt="明天哈哈哈，明天…但是明天好像时间安排又有点错开了所以明天再看吧。明天可能是明天早上。"
                elif [ $infer_spk_name == "tts_Lmand_Sinhouse-multistyle_P1/sinong_conv_1103_update" ];then
                    prompt_wav=/mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/duibiao/sinong_conv_1103_update/wav/sinong_0923_talk_9.wav
                    prompt_txt="谁很男人啊？巨石强森很男人？对，我承认他很男人，但是你们觉得你们能跟，你们能够跟巨石强森比吗？"
                elif [ $infer_spk_name == "tts_Lmand_Sinhouse-multistyle_P1/taozi_conv_1108" ];then
                    prompt_wav=/mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/duibiao/taozi_conv_1108/wav/000222.wav
                    prompt_txt="所以我觉得，这一块儿是不是有可能也会成为，人工智能技术的一个，开展前景，一个具有，呃有一个非常好的发展前景的一个，领域呢？"
                elif [ $infer_spk_name == "tts_Lmand_Sinhouse-multistyle_P1/xudong_yqc_1105" ];then
                    prompt_wav=/mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/duibiao/xudong_yqc_1105/wav/xudong_1103_yqc_003_333.wav
                    prompt_txt="啊这，嗯这个人我跟他相处了一段时间啊我确实觉得，他可能就是这样的习惯吧。"
                elif [ $infer_spk_name == "tts_Lmand_Sinhouse-M392_P1/M392_conv_1121_update" ];then
                    prompt_wav=/mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/duibiao/M392_conv_1121_update/wav/M392_Freetalk_1028_001_8.wav
                    prompt_txt="我觉得可能就是这样，唉其实您说到这其实现在吧好多迪厅的跟之前还真不一样，唉现在迪厅带包厢您知道吗？"
                elif [ $infer_spk_name == "tts_Lmand_Sinhouse-multistyle_P1/maomao_conv_1024_update" ];then
                    prompt_wav=/mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/duibiao/maomao_conv_1024_update/wav/part2_1233.wav
                    prompt_txt="嗯然后它一个是人少一个是可以看极光另外就是，它还有一些什么，就是，你可以看海因为我，很喜欢很喜欢看大海。"
                elif [ $infer_spk_name == "tts_Lmand_Sinhouse-ahu_P1/ahu_conv" ];then
                    prompt_wav=/mnt/bn/huangzhiying-nas-bigtts-data/wenjie/data/duibiao/ahu_conv/wav/ahu_wuxue_1206_talk_01_01_ahu_5.wav
                    prompt_txt="那那吃完之后中午不会饿？唉中午不会太饱了吗？"
                fi

                if [ $prompt_wav != "" ];then
                    cal_asv='TRUE'
                    cp $metalst $metalst.temp_${infer_spk_basename}
                    sed -i "s#|#|${prompt_txt}|${prompt_wav}|#g" $metalst.temp_${infer_spk_basename}
                    metalst=$metalst.temp_${infer_spk_basename}
                fi
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

                if [ $prompt_wav != "" ];then
                    [ -d ${out_wav_dir}/../analyse_asv ] && rm -rf ${out_wav_dir}/../analyse_asv
                    python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/analyse/analyse_asv_listentest_single.py \
                                                                ${out_wav_dir}/../wav_res_ref_text.asv \
                                                                ${out_wav_dir}/../analyse_asv
                    cp $prompt_wav ${out_wav_dir}/../analyse_asv/ref.wav
                fi
            done
        done
    done
fi


# infer seed1996
if [ ${stage} -eq 3 ];then
    hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_streaming/text2semantic/sft_WFVAE_v2_labv3_punc_data_id446_bt14000_16A100_accu5_scr0.0_spFalse_v2
    local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_streaming/text2semantic/sft_WFVAE_v2_labv3_punc_data_id446_bt14000_16A100_accu5_scr0.0_spFalse_v2
    ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints/ | awk '{print $8}' | grep -E '115000'`
    for ar_ckpt_path in $ar_ckpt_paths;do
        step=`echo $ar_ckpt_path | awk -F"/" '{print $NF}' | cut -d "-" -f 2 | cut -d "=" -f 2`
        # for infer_spk_name in tts_Lmand_Sinhouse-multistyle_P1/taozi_conversation tts_Lmand_Sinhouse-multistyle_P1/xudong_conv_1028 tts_Lmand_Sinhouse-multistyle_P1/M392_conv tts_Lmand_Sinhouse-multistyle_P1/maomao_conv_1024_update;do
        # for infer_spk_name in tts_Lmand_Sinhouse-multistyle_P1/sophie_conv_1020;do
        for infer_spk_name in tts_Lmand_Sinhouse-multistyle_P1/xudong_conv_1028;do
            infer_spk_basename=`echo $infer_spk_name | cut -d "/" -f 2`
            tag_id=3
            # for pair in flow_tts_zh-listen_all_new_en_select flow_tts_zh-listen_all_new_codeswitch_select flow_tts_zh-listen_all_new_zh_select;do
            # for pair in flow_tts_zh-listen_all_new_zh_select;do
            for pair in flow_tts_zh-listen_all_new_zh_123;do
                cd $work_dir
                if [[ ! -d ./resource ]]; then
                    hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource
                fi

                testspk=`echo $pair | cut -d "-" -f 1`
                testset=`echo $pair | cut -d "-" -f 2`

                lang='en'
                if [ $pair == "flow_tts_zh-wer_rand20" -o $pair == "flow_tts_zh-yuqici" -o $pair == "flow_tts_zh-temp_demo20231104_v7" -o $pair == "flow_tts_zh-deploy_test" -o $pair == "flow_tts_zh-erhuayin" -o $pair == "flow_tts_zh-yuqici_haha" -o $pair == "flow_tts_zh-listen_all_new_zh_select" -o $pair == "flow_tts_zh-listen_all_new_codeswitch_select" -o $pair == "flow_tts_zh-listen_all_new_zh_123" ]; then
                    lang='zh'
                fi

                cal_asv='FALSE'

                metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}
                out_wav_dir=$local_path/infer/${pair}/step${step}_${infer_spk_basename}_tag_id${tag_id}_use_spFalse_online/wav

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
                    --run_opts.spk2id recipes/text2semantic/datasets/dict/spk2id.parquet.json \
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
                python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/analyse/analyse_wer_listentest_single.py \
                                                                analyse \
                                                                ${out_wav_dir}/../wav_res_ref_text.wer \
                                                                ${out_wav_dir}/.. \
                                                                $lang
            done
        done
    done
fi
