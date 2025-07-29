pip3 install confusables
pip3 install --upgrade attrs;

export PYTHONPATH=$PYTHONPATH:mariana

model_name="stage3_4RVQ_RP"
model_cls="recipes.umm2.modules.stages.stage2.Stage2"
model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/2025071723_UMM-stage3_64GPU_fused_attnmask_model_RVQRP4/checkpoints/step=0080000.ckpt"
# change the output dir to yours
out_dir="./.module_cache/"
nsample=10
inference_R=4
hdfs dfs -get hdfs://harunawl/home/byte_data_seed_wl/speech/user/rui.xia/bigmusic/tokenizer /opt/tiger/tokenizer

tasks="loss,locality,token_repetition"
python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample --tasks $tasks --inference_R $inference_R

tasks="ctc_wer"
python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name \
                                             --model_cls $model_cls \
                                             --model_path $model_path \
                                             --out_dir $out_dir \
                                             --nsample $nsample \
                                             --tasks $tasks \
                                             --inference_R $inference_R