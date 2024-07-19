#!/bin/bash

set -x

work_dir=$(cd $(dirname $0); pwd)/../../../../../
cd $work_dir


[ ! -d resource ] && hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource .

temperature=0.9
thresh=0.9
mode=greedy
seed=1996
eos_weight=1.0

src_lang=zh
tgt_lang=zh
ckpt_path=""
meta_lst=""
output_path=""

use_spk_id=False
speaker_name=""

use_spk_tag=False
tag_id=0

use_offline_splittext=False
offline_splittext_path=

step_out_blank='v2'
max_blank_length=10
max_repeat_times=1

use_text_lang_embedding=False

use_bpe=False
bpe_type="llama"
bpe_dir="resource/models/llama_7B_tokenizer"

use_pre_utt=True

version=merge_v1
precision=fp16
icl_mode=non-continuation

. scripts/parse_options.sh

mkdir -p $output_path

bash launch.sh predict \
        -c apps/bigtts/umm/ar/conf/infer_bigtts_ar_shortform.yaml \
        --pl_module.semantic_precision ${precision} \
        --run_opts.meta_lst $meta_lst \
        --run_opts.umm_ckpt_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.master.ckpt \
        --run_opts.umm_version 0.6.2 \
        --run_opts.semantic_model_path $ckpt_path \
        --run_opts.output_path $output_path \
        --run_opts.src_lang $src_lang \
        --run_opts.tgt_lang $tgt_lang \
        --run_opts.temperature $temperature \
        --run_opts.thresh $thresh \
        --run_opts.mode $mode \
        --run_opts.use_offline_splittext $use_offline_splittext \
        --run_opts.offline_splittext_path $offline_splittext_path \
        --run_opts.step_out_blank $step_out_blank \
        --run_opts.max_blank_length $max_blank_length \
        --run_opts.max_repeat_times $max_repeat_times \
        --run_opts.seed $seed \
        --run_opts.version ${version} \
        --run_opts.icl_mode $icl_mode \
        --run_opts.use_spk_id $use_spk_id \
        --run_opts.speaker_name $speaker_name \
        --run_opts.use_spk_tag $use_spk_tag \
        --run_opts.tag_id $tag_id \
        --run_opts.use_bpe $use_bpe \
        --run_opts.bpe_type $bpe_type \
        --run_opts.bpe_dir $bpe_dir \
        --run_opts.use_text_lang_embedding $use_text_lang_embedding \
        --run_opts.use_pre_utt ${use_pre_utt} \
        --pl_module.eos_weight $eos_weight

