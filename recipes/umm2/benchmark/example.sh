pip3 install confusables
pip3 install --upgrade attrs;

export PYTHONPATH=$PYTHONPATH:mariana

# model_name="stage3_dual_decoder_orth0"
model_name="stage3_3RVQ_RP"
model_cls="recipes.umm2.modules.stages.stage2.Stage2"

# RVQ3 RP
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/2025072916_UMM-stage3_64GPU_fused_attnmask_model_RVQRP3_14bit/checkpoints/step=0065000.ckpt"
# Dual decoder
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/lite_fused_rmpad_dual_decoder_SD4_AD12_orthLoss1_separateVQ_semanticEntropy13bit_acousticEntropy15bit_bucket2088k/checkpoints/step=0060000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/lite_fused_rmpad_dual_decoder_SD4_AD12_orthLoss0_separateVQ_semanticEntropy13bit_acousticEntropy15bit_bucket2088k/checkpoints/step=0040000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/lite_fused_rmpad_dual_decoder_SD6_AD12_orthLoss1_separateVQ_semanticPitchChromaCTCEntropy13bit_acousticMelEntropy15bit_bucket1920k/checkpoints/step=0070000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/lite_fused_dual_decoder_SD6_AD12_orth1_sepVQ_FixedSf0crmCTCEntropy14bit_AmelEntropy15bit_bucket1680k/checkpoints/step=0080000.ckpt"
model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/lite_fused_dualdec_SD6AD12_orth1_sep_FixedSf0crmCTCRP13bit_AmelRP2x14bit_bucket1680k/checkpoints/step=0085000.ckpt"

# change the output dir to yours
out_dir="./.module_cache/"
nsample=20
inference_R=2   # 2+1

hdfs dfs -get hdfs://harunawl/home/byte_data_seed_wl/speech/user/rui.xia/bigmusic/tokenizer /opt/tiger/tokenizer

tasks="loss,locality,token_repetition"
python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample --tasks $tasks --inference_R $inference_R

# tasks="ctc_wer"
# python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name \
#                                              --model_cls $model_cls \
#                                              --model_path $model_path \
#                                              --out_dir $out_dir \
#                                              --nsample $nsample \
#                                              --tasks $tasks \
#                                              --inference_R $inference_R