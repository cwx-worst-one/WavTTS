#!/bin/bash

#########################################################################################
############# All configurations can be override like --xx yy from cli. #################
#########################################################################################

# EXAMPLE:
# bash apps/bigtts/umm/ar/infer/infer-umm-ar-spkid-wer_v3.sh \
#     --src_lang en \
#     --tgt_lang en \
#     --seed 1997 \
#     --step 100k \
#     --eos_weight 0.8 \
#     --ckpt_root hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenjiawei \
#     --log_name umm_ar \
#     --log_version test_1 \
#     --model_version merge_v2_1

# And if you wanna execute multiple tests at same time, please specify a different port for
# each run, for example:
# ARNOLD_WORKER_0_PORT=10000 bash apps/bigtts/umm/ar/infer/infer-umm-ar-spkid-wer_v3.sh ...
# ARNOLD_WORKER_0_PORT=10001 bash apps/bigtts/umm/ar/infer/infer-umm-ar-spkid-wer_v3.sh ...
# ARNOLD_WORKER_0_PORT=10003 bash apps/bigtts/umm/ar/infer/infer-umm-ar-spkid-wer_v3.sh ...


############################ DECODING CONFIGURATION ######################################
thresh=0.9
mode=naive
eos_weight=1.0
step_out_blank=v2 # v0(不使用发愣) or v1(5个重复帧发愣) or v2(滑窗发愣) or v3(滑窗发愣+发愣逻辑)
temperature=0.9
max_blank_length=5
max_repeat_times=1

############################ INFERENCE CONFIGURATION #####################################
src_lang=zh
tgt_lang=zh
seed=1997

############################### MODEL CONFIGURATION ######################################
umm_version=v0.6.2
umm_ckpt=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.new.ckpt
model_version="merge_v2" # spkid_v3_3 or merge_v1_1 or merge_v2_1 or merge_v3_1 or merge_v1_1_cfg

############################# CHECKPOINT CONFIGURATION ###################################
# We assume that checkpoint saved in a path as following:
# ${ckpt_root}/${log_name}/${log_version}/checkpoints/xxx.ckpt

ckpt_root=    # hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenjiawei
step=         # 100k, automatically convert 100k to 100000
log_name=     # same as training log name
log_version=  # same as training log version
note=         # test


meta_root=/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset
output_root=/mnt/bn/data-storage/bigtts/ar/output

echo "cmd: $0 $@"

. scripts/parse_options.sh



# check configurations
if [ -z $ckpt_root ]; then
    echo "ckpt_root must be provided, please specify it by --ckpt_root xxx"
    exit 1
fi
if [ -z $step ]; then
    echo "step must be provided, please specify it by --step xxx"
    exit 1
fi

ckpt_path=$(hdfs dfs -ls ${ckpt_root}/${log_name}/${log_version}/checkpoints/ | grep ${step//k/000} | awk '{print $NF}')

if [ -z $ckpt_path ]; then
    echo "No ckpt found"
    echo "  ckpt_root=${ckpt_root}"
    echo "  log_name=${log_name}"
    echo "  log_version=${log_version}"
    echo "  step=${step}"
    exit 1
fi

if [ $step_out_blank == "v2" ]; then
    temperature=0.5
    max_blank_length=$(awk -v temp=$temperature 'BEGIN{print int(5 / temp)}')
elif [ $step_out_blank == "v3" ]; then
    temperature=0.9
    max_blank_length=$(awk -v temp=$temperature 'BEGIN{print int(5 / temp)}')
    max_repeat_times=2
else
    max_blank_length=5
    temperature=0.9
fi

if [ $src_lang = $tgt_lang ]; then
    icl_mode="continuation"
else
    icl_mode="non-continuation"
fi


suffix=${model_version}_${icl_mode}_blank${step_out_blank}_win${max_blank_length}_200wh_eos${eos_weight}_1pad1_copy3_wer_seed$seed

if [ $src_lang == "en" ] && [ $tgt_lang == "en" ]; then
    save_npy_name=umm_t${temperature}_p${thresh}_ar${step}_${tgt_lang}_${umm_version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/${tgt_lang}/meta.lst

elif [ $src_lang == "zh" ] && [ $tgt_lang == "zh" ]; then
    save_npy_name=umm_t${temperature}_p${thresh}_ar${step}_${tgt_lang}_${umm_version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/${tgt_lang}/meta.lst

elif [ $src_lang == "en" ] && [ $tgt_lang == "zh" ]; then
    save_npy_name=umm_t${temperature}_p${thresh}_ar${step}_${src_lang}2${tgt_lang}_${umm_version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst

elif [ $src_lang == "zh" ] && [ $tgt_lang == "en" ]; then
    save_npy_name=umm_t${temperature}_p${thresh}_ar${step}_${src_lang}2${tgt_lang}_${umm_version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst

### another en2zh
elif [ $src_lang == "zh_en" ] && [ $tgt_lang == "zh" ]; then
    save_npy_name=umm_t${temperature}_p${thresh}_ar${step}_${src_lang}2${tgt_lang}_${umm_version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/en2${tgt_lang}_meta.lst
    tgt_lang=$src_lang

### another zh2en
elif [ $src_lang == "zh_en" ] && [ $tgt_lang == "en" ]; then
    save_npy_name=umm_t${temperature}_p${thresh}_ar${step}_${src_lang}2${tgt_lang}_${umm_version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/zh2${tgt_lang}_meta.lst
    tgt_lang=$src_lang

else
    echo "Unsupported Language!"
    exit
fi

output_path=${output_root}/${save_npy_name}/
mkdir -p $output_path

echo "CONFIGURATION:"
echo "    thresh=$thresh"
echo "    mode=$mode"
echo "    eos_weight=$eos_weight"
echo "    step_out_blank=$step_out_blank"
echo "    max_blank_length=$max_blank_length"
echo "    max_repeat_times=$max_repeat_times"
echo "    temperature=$temperature"

echo "    src_lang=$src_lang"
echo "    tgt_lang=$tgt_lang"
echo "    seed=$seed"
echo "    umm_version=$umm_version"
echo "    model_version=$model_version"

echo "    ckpt_root=$ckpt_root"
echo "    step=$step"
echo "    log_name=$log_name"
echo "    log_version=$log_version"
echo "    ckpt=$ckpt_path"

echo "    icl_mode=$icl_mode"
echo "    note=$note"
echo "    meta_root=$meta_root"
echo "    output_path=$output_path"
echo "    meta_lst=$meta_lst"


TORCHRUN samantha/main.py predict \
    -c apps/bigtts/umm/ar/conf/infer_bigtts_ar.yaml \
    --pl_module.semantic_precision fp16 \
    --run_opts.num_workers 1 \
    --run_opts.meta_lst $meta_lst \
    --run_opts.umm_ckpt_path ${umm_ckpt} \
    --run_opts.semantic_model_path $ckpt_path \
    --run_opts.output_path $output_path \
    --run_opts.src_lang $src_lang \
    --run_opts.tgt_lang $tgt_lang \
    --run_opts.temperature $temperature \
    --run_opts.thresh $thresh \
    --run_opts.mode $mode \
    --run_opts.step_out_blank $step_out_blank \
    --run_opts.max_blank_length $max_blank_length \
    --run_opts.max_repeat_times $max_repeat_times \
    --run_opts.seed $seed \
    --run_opts.version $model_version \
    --run_opts.icl_mode $icl_mode \
    --pl_module.eos_weight $eos_weight || exit 1

echo output_path=$output_path
