#!/bin/bash

echo ">>>>> Start Inference"

ar_base_dir=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenkuan
ar_project_name=sft_semantic_model_0.7B
ar_exp_name=data_id3575_bz4_accu2_tagTrue03v11-cln_textlang_bpeTrue0.1_pre180k_resetopt
ar_hdfs_path=$ar_base_dir/$ar_project_name/$ar_exp_name
ar_step=005000
infer_spk_name=tts_Lmand_Sinhouse-taozi_conv_P5/taozi_conv_talk_0124
pair=flow_tts_zh-listen_all_new_en_select  # flow_tts_zh-listen_all_new_en_select flow_tts_zh-listen_all_new_zh_select_v2

src_lang="zh"
tgt_lang="en"
tag_id=2
sample_mode="greedy"
temperature=0.9
only_use_global_prompt=True  # False for tgt lang zh, True for en

diffusion_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/consistency_distill_fixed/CD_block32_8H800_setting0_UMM062_UMMnoDrop_LR8e-6_guid1to6_tDiff0.02_X0Prob0.1_L2SSIM
diffusion_hdfs_path=`hdfs dfs -ls $diffusion_hdfs_dir/checkpoints/ | awk '{print $8}' | grep -E '300000'`
diffusion_step=`echo $diffusion_hdfs_path | awk -F"/" '{print $NF}' | cut -d "-" -f 2 | cut -d "=" -f 2`

out_dir=/mnt/bn/chenkuan-nas-lq-001/outputs/code_merge/master

port_index=0

. scripts/parse_options.sh

out_dir=$out_dir/$ar_project_name/$ar_exp_name


echo ">>>>> Start AR Inference"
ar_ckpt_path=`hdfs dfs -ls $ar_hdfs_path/checkpoints/ | awk '{print $8}' | grep -E $ar_step`
infer_spk_basename=`echo $infer_spk_name | cut -d "/" -f 2`
testspk=`echo $pair | cut -d "-" -f 1`
testset=`echo $pair | cut -d "-" -f 2`

# meta_lst for infer。注意这里是按照固定的格式来命名meta_lst
meta_lst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}.temp_${infer_spk_basename}

# [optional] 外部提供切句信息，此时就不会调用切句服务进行切句，主要是为了保证每次合成的切句结果是一样的
offline_splittext_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}.split_online
use_offline_splittext=True
if [ ! -f $offline_splittext_path ];then
    use_offline_splittext=False
fi

out_umm_dir=$out_dir/infer/${pair}/shortform-causal_lang${src_lang}-${tgt_lang}_astep${ar_step}_dstep${diffusion_step}_${infer_spk_basename}_tag${tag_id}_${sample_mode}-${temperature}_b10_global${only_use_global_prompt}/umm
[ ! -d $out_umm_dir ] && mkdir -p $out_umm_dir
[ ! -f $out_umm_dir/../`basename $meta_lst` ] && cp $meta_lst $out_umm_dir/../

ARNOLD_WORKER_0_PORT=`expr 10000 + ${port_index}` bash apps/bigtts/umm/ar/scripts/infer-umm-ar-wer_inhouse.sh \
    --src_lang $src_lang \
    --tgt_lang $tgt_lang \
    --ckpt_path $ar_ckpt_path \
    --meta_lst $meta_lst \
    --output_path $out_umm_dir \
    --use_spk_id True \
    --speaker_name $infer_spk_name \
    --use_spk_tag True \
    --tag_id $tag_id \
    --use_offline_splittext $use_offline_splittext \
    --offline_splittext_path $offline_splittext_path \
    --step_out_blank "v2" \
    --max_blank_length 10 \
    --use_text_lang_embedding True \
    --use_bpe True \
    --bpe_type "llama" \
    --bpe_dir "resource/models/llama_7B_tokenizer" \
    --version "merge_v1" \
    --mode $sample_mode \
    --temperature $temperature
echo ">>>>> END AR Inference"

echo ">>>>> Start Diffusion Inference"
cat $out_umm_dir/../meta_split.lst | sort | uniq > $out_umm_dir/../meta_split.lst.sort.uniq
mv $out_umm_dir/../meta_split.lst.sort.uniq $out_umm_dir/../meta_split.lst
meta_lst=$out_umm_dir/../meta_split.lst

            
out_wav_dir=$out_dir/infer/${pair}/shortform-causal_lang${src_lang}-${tgt_lang}_astep${ar_step}_dstep${diffusion_step}_${infer_spk_basename}_tag${tag_id}_${sample_mode}-${temperature}_b10_global${only_use_global_prompt}/wav
prompt_wav_dir=/mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/duibiao/${infer_spk_name}/wav

ARNOLD_WORKER_0_PORT=`expr 20000 + ${port_index}` bash apps/bigtts/umm/diffusion/scripts/ar-diffusion-vocoder-wer_inhouse_streaming.sh \
    --src_lang $src_lang \
    --tgt_lang $tgt_lang \
    --prompt_wav_dir $prompt_wav_dir \
    --meta_lst $meta_lst \
    --out_umm_dir $out_umm_dir \
    --out_wav_dir $out_wav_dir/../wav_split \
    --diffusion_ckpt_path $diffusion_hdfs_path \
    --only_use_global_prompt $only_use_global_prompt
echo ">>>>> End Diffusion Inference"

echo ">>>>> Start Merge&Norm Wav"
python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_umm_shortform/recipes/bigmusic/utils/merge_split_wavs.py $meta_lst ${out_wav_dir}_split $out_wav_dir
python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_streaming/recipes/text2semantic/utils/norm_vol1.0.py ${out_wav_dir} ${out_wav_dir}_vol1.0
echo ">>>>> End Merge&Norm Wav "

echo ">>>>> Start Eval "
bigtts_eval_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval
meta_lst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/${testspk}/meta.lst.${testset}.temp_${infer_spk_basename}
python3 ${bigtts_eval_dir}/utils/get_wav_res_ref_text.py $meta_lst ${out_wav_dir} ${out_wav_dir}/../wav_res_ref_text
echo ${out_wav_dir}/../wav_res_ref_text
bash ${bigtts_eval_dir}/run-eval_parse.sh \
    --bigtts_eval_dir ${bigtts_eval_dir} \
    --asr_type internal \
    --infile ${out_wav_dir}/../wav_res_ref_text \
    --outdir ${out_wav_dir}/../ \
    --lang ${tgt_lang} \
    --cal_asv 'TRUE'

[ -d ${out_wav_dir}/../analyse ] && rm -rf ${out_wav_dir}/../analyse
python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/analyse/analyse_wer_listentest_single.py \
    analyse \
    ${out_wav_dir}/../wav_res_ref_text.wer \
    ${out_wav_dir}/.. \
    $tgt_lang


[ -d ${out_wav_dir}/../analyse_asv ] && rm -rf ${out_wav_dir}/../analyse_asv
python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts-eval/analyse/analyse_asv_listentest_single.py \
    ${out_wav_dir}/../wav_res_ref_text.asv \
    ${out_wav_dir}/../analyse_asv
echo ">>>>> End Eval "