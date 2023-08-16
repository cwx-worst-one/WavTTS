set -x

stage=1
batch_total_tokens=6000 #17500
mode=debug  # train or debug
dataname=ll-rp-fq-others

. /opt/tiger/samantha/recipes/text2semantic/utils/parse_options.sh

if [ $mode == "debug" ]; then
    work_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge
fi

if [ $mode == "train" ]; then
    work_dir=/opt/tiger/samantha
fi

if [ ${stage} -eq 1 ];then
    log_name=pretrain_labv3_punc
    version=${dataname}_bt${batch_total_tokens}_32A100_accu2_stop
    hdfs_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/huangzhiying.92/exp/samantha_bigtts_merge/text2semantic/${log_name}_${version}

    cd $work_dir
    urls_path=recipes/text2semantic/datasets/urls_lst/url.lst.WFVAE_v2_labv3_punc.${dataname}
    urls=`cat $urls_path`
    urls=`echo $urls` # "\n" => " "
    # urls="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc/11labs/*/chunk*/*.tar hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc/duibiao/*/chunk*/*.tar"
    ckpt_path=$hdfs_path/checkpoints/last.ckpt
    bash launch.sh fit --config recipes/text2semantic/conf/llama/vae_llama_ctiga_wds_stop.yaml \
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
                        --trainer.accumulate_grad_batches 2 \
                        --run_opts.learning_rate 0.0003 \
                        --scheduler_cls.cycle_steps 400000 \
                        --trainer.log_every_n_steps 20 \
                        --run_opts.metaid_to_textid 'recipes/text2semantic/datasets/dict/metaid_to_textid.v2.en_zh.json' \
                        --run_opts.textid_version 'v2' \
                        --run_opts.weight_decay 0.1 \
                        --llama_config.attn_pdrop 0.0 \
                        --llama_config.resid_pdrop 0.0 \
                        --llama_config.dim 1536 \
                        --llama_config.n_layers 30 \
                        --run_opts.use_phoneme_loss True \
                        --ckpt_path $ckpt_path
fi


# infer seed1996
if [ ${stage} -eq 2 ];then
    hdfs_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/huangzhiying.92/exp/samantha_bigtts_merge/text2semantic/pretrain_labv3_punc_ll-rp-fq-others_bt17500_32A100_accu2_stop
    local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_merge/text2semantic/pretrain_labv3_punc_ll-rp-fq-others_bt17500_32A100_accu2_stop
    ar_ckpt_path=$hdfs_path/checkpoints/last.ckpt
    step=last

    for pair in inner_testset_en-prompt1 out_testset_en-prompt1 inner_testset_zh-prompt1 out_testset_zh-prompt1;do
        cd $work_dir
        testspk=`echo $pair | cut -d "-" -f 1`
        testset=`echo $pair | cut -d "-" -f 2`

        lang='en'
        if [ $testspk == "inner_testset_zh" -o $testspk == "out_testset_zh" ]; then
            lang='zh'
        fi

        out_bn_dir=$local_path/infer/${pair}/step${step}_seed1996/bn
        [ ! -d $out_bn_dir ] && mkdir -p $out_bn_dir

        metalst_WFVAE_LZX=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/testset/${testspk}/meta.lst.${testset}
        timestamp=$(date +%s)
        thread_dir=/tmp/thread_metas_$timestamp/
        sudo mkdir $thread_dir
        num_job=$ARNOLD_WORKER_GPU
        num=`wc -l $metalst_WFVAE_LZX | awk -F' ' '{print $1}'`
        num_per_thread=`expr $num / $num_job + 1`
        sudo split -l $num_per_thread --additional-suffix=.lst -d $metalst_WFVAE_LZX $thread_dir/thread-

        prompt_tacolab_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/testset/${testspk}/lab_newv3_punc_wfvae_lzx_v2/prompt_tacolabs
        infer_tacolab_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/testset/${testspk}/lab_newv3_punc_wfvae_lzx_v2/tacolabs 
        prompt_bn_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/testset/${testspk}/lab_newv3_punc_wfvae_lzx_v2/prompt_WFVAE_LZX
        for rank in $(seq 0 $((num_job - 1)));do
        {
            python3 recipes/text2semantic/scripts/llama/infer_text2semantic_vae_raw_stop.py \
            --ar_ckpt_path=$ar_ckpt_path \
            --meta_file=$thread_dir/thread-0$rank.lst \
            --device=cuda:$rank \
            --out_dir=$out_bn_dir \
            --prompt_tacolab_dir=$prompt_tacolab_dir \
            --infer_tacolab_dir=$infer_tacolab_dir \
            --prompt_bn_dir=$prompt_bn_dir \
            --lab2id_path=recipes/text2semantic/datasets/dict/metaid_to_textid.v2.en_zh.json \
            --seed 1996
        }&
        done
        wait

        out_wav_dir=$local_path/infer/${pair}/step${step}_seed1996/wav

        cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/VIFSpeech_original
        python3 bn2wav.py -c configs/multi_speaker.json -m CLONE -l test -p G.pth \
                        --input_dir $out_bn_dir \
                        --output_dir $out_wav_dir \
                        --device cuda:0
        cd -

        cd $work_dir
        asr_type=internal
        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/testset/${testspk}/meta.lst.${testset}
        python3 recipes/text2semantic/utils/bigtts-eval/utils/get_wav_res_ref_text.py $metalst ${out_wav_dir} ${out_wav_dir}/../wav_res_ref_text
        bigtts_eval_dir=$work_dir/recipes/text2semantic/utils/bigtts-eval
        bash $work_dir/recipes/text2semantic/utils/bigtts-eval/run-eval_parse.sh \
                                                        --bigtts_eval_dir $bigtts_eval_dir \
                                                        --asr_type $asr_type \
                                                        --infile ${out_wav_dir}/../wav_res_ref_text \
                                                        --outdir ${out_wav_dir}/../ \
                                                        --lang $lang \
                                                        --cal_asv "TRUE"
        cd -
    done
fi


# select checkpoint seed1996
if [ ${stage} -eq 3 ];then
    hdfs_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/huangzhiying.92/exp/samantha_bigtts_merge/text2semantic/pretrain_labv3_punc_ll-rp-fq-others_bt17500_32A100_accu2_stop
    local_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_merge/text2semantic/pretrain_labv3_punc_ll-rp-fq-others_bt17500_32A100_accu2_stop

    ar_ckpt_paths=`hdfs dfs -ls $hdfs_path/checkpoints | awk '{print $NF}' | grep -v item | grep -v last | awk 'NR%1==0'`

    for ar_ckpt_path in $ar_ckpt_paths;do
        step=`echo $ar_ckpt_path | awk -F"/" '{print $NF}' | cut -d "-" -f 2 | cut -d "=" -f 2`
        for pair in inner_testset_en-prompt1 out_testset_en-prompt1 inner_testset_zh-prompt1 out_testset_zh-prompt1;do
            cd $work_dir
            testspk=`echo $pair | cut -d "-" -f 1`
            testset=`echo $pair | cut -d "-" -f 2`

            lang='en'
            if [ $testspk == "inner_testset_zh" -o $testspk == "out_testset_zh" ]; then
                lang='zh'
            fi

            out_bn_dir=$local_path/infer/${testspk}/step${step}_seed1996/bn
            [ -d $out_bn_dir ] && continue
            [ ! -d $out_bn_dir ] && mkdir -p $out_bn_dir

            metalst_WFVAE_LZX=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/testset/${testspk}/meta.lst.${testset}
            timestamp=$(date +%s)
            thread_dir=/tmp/thread_metas_$timestamp/
            sudo mkdir $thread_dir
            num_job=$ARNOLD_WORKER_GPU
            num=`wc -l $metalst_WFVAE_LZX | awk -F' ' '{print $1}'`
            num_per_thread=`expr $num / $num_job + 1`
            sudo split -l $num_per_thread --additional-suffix=.lst -d $metalst_WFVAE_LZX $thread_dir/thread-

            prompt_tacolab_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/testset/${testspk}/lab_newv3_punc_wfvae_lzx_v2/prompt_tacolabs
            infer_tacolab_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/testset/${testspk}/lab_newv3_punc_wfvae_lzx_v2/tacolabs
            prompt_bn_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/testset/${testspk}/lab_newv3_punc_wfvae_lzx_v2/prompt_WFVAE_LZX
            for rank in $(seq 0 $((num_job - 1)));do
            {
                python3 recipes/text2semantic/scripts/llama/infer_text2semantic_vae_raw.py \
                --ar_ckpt_path=$ar_ckpt_path \
                --meta_file=$thread_dir/thread-0$rank.lst \
                --device=cuda:$rank \
                --out_dir=$out_bn_dir \
                --prompt_tacolab_dir=$prompt_tacolab_dir \
                --infer_tacolab_dir=$infer_tacolab_dir \
                --prompt_bn_dir=$prompt_bn_dir \
                --lab2id_path=recipes/text2semantic/datasets/dict/metaid_to_textid.v2.en_zh.json \
                --seed 1996
            }&
            done
            wait

            out_wav_dir=$local_path/infer/${testspk}/step${step}_seed1996/wav

            cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/VIFSpeech_original
            python3 bn2wav.py -c configs/multi_speaker.json -m CLONE -l test -p G.pth \
                            --input_dir $out_bn_dir \
                            --output_dir $out_wav_dir \
                            --device cuda:0
            cd -

            cd $work_dir
            asr_type=internal
            metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/testset/${testspk}/meta.lst.${testset}
            python3 recipes/text2semantic/utils/bigtts-eval/utils/get_wav_res_ref_text.py $metalst ${out_wav_dir} ${out_wav_dir}/../wav_res_ref_text
            bigtts_eval_dir=$work_dir/recipes/text2semantic/utils/bigtts-eval
            bash $work_dir/recipes/text2semantic/utils/bigtts-eval/run-eval_parse.sh \
                                                            --bigtts_eval_dir $bigtts_eval_dir \
                                                            --asr_type $asr_type \
                                                            --infile ${out_wav_dir}/../wav_res_ref_text \
                                                            --outdir ${out_wav_dir}/../ \
                                                            --lang $lang \
                                                            --cal_asv "TRUE"
            cd -
        done
    done
fi
