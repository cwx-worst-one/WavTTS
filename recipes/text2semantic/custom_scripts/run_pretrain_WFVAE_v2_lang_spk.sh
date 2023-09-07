set -x

stage=1
batch_total_tokens=17500
mode=debug  # train or debug
dataname=ll-rp-fq-others

. /opt/tiger/samantha/scripts/parse_options.sh

if [ $mode == "debug" ]; then
    work_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_mix_merge
fi

if [ $mode == "train" ]; then
    work_dir=/opt/tiger/samantha
fi

if [ ${stage} -eq 1 ];then
    log_name=pretrain_labv3_punc
    version=${dataname}_bt${batch_total_tokens}_32A100_lang_spk
    hdfs_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/huangzhiying.92/exp/samantha_bigtts_mix_merge/text2semantic/${log_name}_${version}

    cd $work_dir
    urls_path=recipes/text2semantic/datasets/urls_lst/url.lst.WFVAE_v2_labv3_punc.${dataname}
    urls=`cat $urls_path`
    urls=`echo $urls` # "\n" => " "
    # urls="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc_1879h/11labs/*/chunk*/*.tar"
    # urls="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc/11labs/*/chunk*/*.tar hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc/duibiao/*/chunk*/*.tar"
    bash launch.sh fit --config recipes/text2semantic/conf/llama/vae_llama_ctiga_wds_lang_spk.yaml \
                        --run_opts.urls $urls \
                        --run_opts.hdfs_path $hdfs_path \
                        --run_opts.return_full_seq False \
                        --run_opts.log_dir ./logs \
                        --run_opts.log_name ${log_name} \
                        --run_opts.version ${version} \
                        --run_opts.batch_total_tokens ${batch_total_tokens} \
                        --run_opts.precision bf16 \
                        --run_opts.checkpointing False \
                        --run_opts.n_step_save 5000 \
                        --trainer.accumulate_grad_batches 1 \
                        --run_opts.learning_rate 0.0003 \
                        --scheduler_cls.cycle_steps 400000 \
                        --trainer.log_every_n_steps 20 \
                        --run_opts.weight_decay 0.1 \
                        --llama_config.attn_pdrop 0.0 \
                        --llama_config.resid_pdrop 0.0 \
                        --llama_config.dim 1536 \
                        --llama_config.n_layers 30 \
                        --run_opts.use_lang_id True \
                        --run_opts.lang_tokens_num 200 \
                        --run_opts.lang2id recipes/text2semantic/datasets/dict/lang2id.json \
                        --run_opts.strategy ddp_find_unused_parameters_true
fi


# infer seed1996
if [ ${stage} -eq 2 ];then
    hdfs_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/huangzhiying.92/exp/samantha_bigtts_mix/text2semantic/pretrain_labv3_punc_ll-rp-fq-others_bt17500_32A100_lang_spk_pure
    local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_mix/text2semantic/pretrain_labv3_punc_ll-rp-fq-others_bt17500_32A100_lang_spk_pure
    ar_ckpt_path=$hdfs_path/checkpoints/epoch=00-step=110000-kl_loss=0.61.ckpt
    step=`echo $ar_ckpt_path | awk -F"/" '{print $NF}' | cut -d "-" -f 2 | cut -d "=" -f 2`

    # for pair in inner_testset_en-prompt1 out_testset_en-prompt1 inner_testset_zh-prompt1 out_testset_zh-prompt1;do
    # for pair in temp-zh_0703_F071 temp-zh_0703_Fmaomao temp-zh_0703_M174 temp-zh_0703_M656;do
    # for pair in cross_lingual_en_zh-SAMI_makeup cross_lingual_zh_en-SAMI_zh_xiaoqin;do
    # for pair in inner_testset_en-prompt1_rand200 out_testset_en-prompt1_rand200 inner_testset_zh-prompt1_rand200 out_testset_zh-prompt1_rand200 inner_testset_en2zh-prompt1_rand200 inner_testset_zh2en-prompt1_rand200 out_testset_en2zh-prompt1_rand200 out_testset_zh2en-prompt1_rand200;do
    # for pair in inner_testset_en-prompt1 out_testset_en-prompt1 inner_testset_zh-prompt1 out_testset_zh-prompt1 inner_testset_en2zh-prompt1 inner_testset_zh2en-prompt1 out_testset_en2zh-prompt1 out_testset_zh2en-prompt1;do
    # for pair in icl_testset_en_1000-hard_500 icl_testset_en_1000-long_100 icl_testset_en_1000-common_2000;do
    # for pair in compare_with_vallex-en2zh_copy;do
    for pair in inner_testset_en2zh-prompt1_rand200;do
        cd $work_dir
        testspk=`echo $pair | cut -d "-" -f 1`
        testset=`echo $pair | cut -d "-" -f 2`

        lang='en'
        if [ $testspk == "inner_testset_zh" -o $testspk == "out_testset_zh" -o $testspk == "temp" -o $testspk == "cross_lingual_en_zh" -o $testspk == "inner_testset_en2zh" -o $testspk == "out_testset_en2zh" -o $testspk == "201_zh2en" ]; then
            lang='zh'
        fi

        cal_asv='TRUE'
        # if [ $testspk == "temp" ]; then
        #     cal_asv='False'
        # fi

        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}
        out_wav_dir=$local_path/infer/${pair}/step${step}_seed1996/wav
        # [ -d $out_wav_dir ] && continue

        prompt_tacolab_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/prompt_lab_newv3_punc
        infer_tacolab_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/infer_lab_newv3_punc

        bash launch.sh predict \
            -c recipes/text2semantic/conf/llama/inference_wvae_icl_lang_spk.yaml \
            --run_opts.meta_lst $metalst \
            --run_opts.ckpt_path $ar_ckpt_path \
            --run_opts.output_dir $out_wav_dir \
            --run_opts.wvae_encoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae/wavevae_encoder_%d.pt \
            --run_opts.wvae_decoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae/wavevae_decoder_%d.pt \
            --run_opts.prompt_tacolab_dir $prompt_tacolab_dir \
            --run_opts.infer_tacolab_dir $infer_tacolab_dir \
            --run_opts.ar_model_name 'VAET2SLangSpkModule' \
            --run_opts.use_lang_id True \
            --run_opts.lang_tokens_num 200 \
            --run_opts.lang2id 'recipes/text2semantic/datasets/dict/lang2id.json'

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
fi


# select checkpoint seed1996
if [ ${stage} -eq 3 ];then
    hdfs_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/huangzhiying.92/exp/samantha_bigtts_mix_merge/text2semantic/pretrain_labv3_punc_ll-rp-fq-others_bt17500_32A100_langid_multidimv3
    local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_mix_merge/text2semantic/pretrain_labv3_punc_ll-rp-fq-others_bt17500_32A100_langid_multidimv3
    ar_ckpt_path=$hdfs_path/checkpoints/epoch=00-step=155000-kl_loss=0.60.ckpt #epoch=00-step=95000-kl_loss=0.66.ckpt
    step=`echo $ar_ckpt_path | awk -F"/" '{print $NF}' | cut -d "-" -f 2 | cut -d "=" -f 2`

    # for pair in inner_testset_en-prompt1 out_testset_en-prompt1 inner_testset_zh-prompt1 out_testset_zh-prompt1;do
    # for pair in temp-zh_0703_F071 temp-zh_0703_Fmaomao temp-zh_0703_M174 temp-zh_0703_M656;do
    # for pair in cross_lingual_en_zh-SAMI_makeup cross_lingual_zh_en-SAMI_zh_xiaoqin;do
    # for pair in inner_testset_en-prompt1_rand200 out_testset_en-prompt1_rand200 inner_testset_zh-prompt1_rand200 out_testset_zh-prompt1_rand200 inner_testset_en2zh-prompt1_rand200 inner_testset_zh2en-prompt1_rand200 out_testset_en2zh-prompt1_rand200 out_testset_zh2en-prompt1_rand200;do
    # for pair in inner_testset_en-prompt1 out_testset_en-prompt1 inner_testset_zh-prompt1 out_testset_zh-prompt1 inner_testset_en2zh-prompt1 inner_testset_zh2en-prompt1 out_testset_en2zh-prompt1 out_testset_zh2en-prompt1;do
    # for pair in icl_testset_en_1000-hard_500 icl_testset_en_1000-long_100 icl_testset_en_1000-common_2000;do
    # for pair in compare_with_vallex-en2zh_betterstudio_copy;do
    for pair in compare_with_vallex-en2zh_debug590_copy;do
        cd $work_dir
        testspk=`echo $pair | cut -d "-" -f 1`
        testset=`echo $pair | cut -d "-" -f 2`

        lang='en'
        if [ $testspk == "inner_testset_zh" -o $testspk == "out_testset_zh" -o $testspk == "temp" -o $testspk == "cross_lingual_en_zh" -o $testspk == "inner_testset_en2zh" -o $testspk == "out_testset_en2zh" -o $testspk == "201_zh2en" ]; then
            lang='zh'
        fi

        cal_asv='TRUE'
        # if [ $testspk == "temp" ]; then
        #     cal_asv='False'
        # fi

        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}
        out_wav_dir=$local_path/infer/${pair}/step${step}_seed1996/wav
        # [ -d $out_wav_dir ] && continue

        prompt_tacolab_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/prompt_lab_newv3_punc
        infer_tacolab_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/infer_lab_newv3_punc

        bash launch.sh predict \
            -c recipes/text2semantic/conf/llama/inference_wvae_icl_langv2_multidimv3.yaml \
            --run_opts.meta_lst $metalst \
            --run_opts.ckpt_path $ar_ckpt_path \
            --run_opts.output_dir $out_wav_dir \
            --run_opts.wvae_encoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae/wavevae_encoder_%d.pt \
            --run_opts.wvae_decoder hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae/wavevae_decoder_%d.pt \
            --run_opts.prompt_tacolab_dir $prompt_tacolab_dir \
            --run_opts.infer_tacolab_dir $infer_tacolab_dir \
            --run_opts.ar_model_name 'VAET2SStopLangv2MultidimModule' \
            --run_opts.use_langid True \
            --run_opts.lang_tokens_num 200 \
            --run_opts.lang2id 'recipes/text2semantic/datasets/dict/lang2id.json'

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
fi
