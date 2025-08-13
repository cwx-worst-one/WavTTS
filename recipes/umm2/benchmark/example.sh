pip3 install confusables
pip3 install --upgrade attrs;

export PYTHONPATH=$PYTHONPATH:mariana

model_name="stage3_dual_decoder_orth1_60k"
model_cls="recipes.umm2.modules.stages.stage2.Stage2"
model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/lite_fused_rmpad_dual_decoder_SD4_AD12_orthLoss1_separateVQ_semanticEntropy13bit_acousticEntropy15bit_bucket2088k/checkpoints/step=0060000.ckpt"
# change the output dir to yours
out_dir="./.module_cache/"
nsample=750
inference_R=2
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