export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
pip3 install confusables
pip3 uninstall attrs attr -y
pip3 install attrs

model_name="debug2"
model_cls="recipes.umm2.modules.stages.stage2.Stage2"

# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage2_rmpad_v4/2025080400_UMM-stage2_Dur5-60_causal_GB15X64H20/checkpoints/"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage2_rmpad_v4/2025080506_UMM-stage2_Dur5-60_causal_GB15X64H20/checkpoints"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage2_rmpad_v4/2025081712_UMM-stage2_Dur5-60_causal_GB15X64H20/checkpoints"
model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage2_rmpad_v4/2025082211_UMM-stage2_Dur5-60_causal_GB15X64H20/checkpoints"
tasks="ctc_wer"
slice_modes='slice'
slice_dur=60
out_dir="./"
nsample=750


python3 recipes/umm2/benchmark/run_eval_stage2.py \
        --model_name $model_name \
        --model_cls $model_cls \
        --model_path $model_path \
        --out_dir $out_dir\
        --nsample $nsample \
        --slice_modes $slice_modes \
        --slice_dur $slice_dur \
        --tasks $tasks \
        --start 50000 \
        --end 150000