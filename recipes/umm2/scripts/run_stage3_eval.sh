pip3 install confusables
pip3 install --upgrade attrs;

export PYTHONPATH=$PYTHONPATH:mariana

model_name="stage3_VQ_baseline"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage3_rmpad_v2/2025070400_UMM-stage3_64GPU_16mins_fused_no_consistencyloss_model/checkpoints/step=0080000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage3_rmpad_v2/2025082113_UMM-stage3_64GPU_12mins_causal_conformer_model/checkpoints/step=0010000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage3_rmpad_v2/2025082202_UMM-stage3_64GPU_12mins_causal_conformer_model/checkpoints/step=0005000.ckpt"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage3_rmpad_v2/2025082216_UMM-stage3_64GPU_12mins_causal_conformer_model/checkpoints/step=0015000.ckpt"
model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage3_rmpad_v2/2025082312_UMM-stage3_64GPU_12mins_causal_conformer_model/checkpoints/step=0030000.ckpt"

model_cls="recipes.umm2.modules.stages.stage2.Stage2"
# change the output dir to yours
out_dir="./"
nsample=750
slice_modes="even"
slice_dur=60
hdfs dfs -get hdfs://harunawl/home/byte_data_seed_wl/speech/user/rui.xia/bigmusic/tokenizer /opt/tiger/tokenizer

tasks="loss,locality"
python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name \
                                            --model_cls $model_cls \
                                            --model_path $model_path \
                                            --out_dir $out_dir \
                                            --nsample $nsample \
                                            --tasks $tasks \
                                            --slice_modes $slice_modes \
                                            --slice_dur $slice_dur

nsample=100
slice_modes='even'
tasks="ctc_wer"
python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name \
                                             --model_cls $model_cls \
                                             --model_path $model_path \
                                             --out_dir $out_dir \
                                             --nsample $nsample \
                                             --tasks $tasks \
                                             --slice_modes $slice_modes