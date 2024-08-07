#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:/mnt/bn/music-ai-unified-repo-lq/yangbing/workspace/samantha
export CUDA_VISIBLE_DEVICES=0

seed=None
src_lang=en
tgt_lang=en

use_spk_tag=True
use_spk_id=True
speaker_name="tts_Lmand_Sinhouse-taozi_conv_P5/taozi_conv_talk_0124"

meta_lst="/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/flow_tts_zh/meta.lst.listen_all_new_en_select_prompten.temp_taozi_conv_talk_0124"
offline_splittext_path="/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/flow_tts_zh/meta.lst.listen_all_new_en_select_prompten.split_online"
ar_ckpt_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenkuan/sft_semantic_model_0.7B/data_id3575_bz4_accu2_tagTrue03v11-cln_textlang_bpeTrue0.1_pre180k_resetopt/checkpoints/step=005000-accu=31.43.ckpt"

only_use_global_prompt=False
diffusion_ckpt_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/consistency_distill_fixed/CD_block32_8H800_setting0_UMM062_UMMnoDrop_LR8e-6_guid1to6_tDiff0.02_X0Prob0.1_L2SSIM/checkpoints/epoch=00-step=300000-loss=0.06.ckpt"


use_offline_splittext=True
if [ ! -f $offline_splittext_path ];then
    use_offline_splittext=False
fi

# chinese&codeswitch tag_id=2 ; english tag_id=1
tag_id=2
if [[ "$tgt_lang" == "en" ]]; then
    tag_id=1
    only_use_global_prompt=True
fi

output_path="/mnt/bn/music-ai-unified-repo-lq/yangbing/workspace/valid_outputs/inhouse_output"

bash apps/bigtts/umm/scripts/infer_ar_diffusion_two_stage_inhouse.sh \
    --src_lang $src_lang \
    --tgt_lang $tgt_lang \
    --use_spk_tag $use_spk_tag \
    --use_spk_id $use_spk_id \
    --speaker_name $speaker_name \
    --tag_id $tag_id \
    --ar_ckpt_path $ar_ckpt_path \
    --meta_lst $meta_lst \
    --use_offline_splittext $use_offline_splittext \
    --offline_splittext_path $offline_splittext_path \
    --output_path $output_path 
    