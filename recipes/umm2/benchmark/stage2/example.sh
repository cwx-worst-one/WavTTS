model_name="debug_stage2"

# stage2
# 100Hz
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025//logs/stage2/100Hz_lr2e-4_bs2p4_cyc200k_CTC25Hz_normhead/checkpoints/step=0255000.ckpt"
# 50Hz
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2/50Hz_lr2e-4_cyc200k_bs5_fixall/checkpoints/step=0255000.ckpt"
# 25Hz old data
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2/25Hz_lr2e-4_cyc200k_bs8_old_data/checkpoints/step=0275000.ckpt"
# 25Hz new data
model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage2/25Hz_lr2e-4_cyc200k_bs8_otf_5-60s/checkpoints/step=0280000.ckpt"


model_cls="recipes.umm2.modules.stages.stage2.Stage2"
out_dir="/mnt/bn/music-llm-nas-lq/qinxin/outputs/tokenizer_eval/"
nsample=50
slice_modes='full'
slice_dur=60

tasks="loss,locality,ctc_wer"
python3 recipes/umm2/benchmark/run_eval_stage2.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample --tasks $tasks --slice_modes $slice_modes --slice_dur $slice_dur

# tasks="ctc_wer"
# python3 recipes/umm2/benchmark/run_eval_stage2.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample --tasks $tasks 

