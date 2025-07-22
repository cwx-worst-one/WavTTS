# model_name="stage3_FT25Hz_VQ_RP_50Hz_CTC25Hz_1x32768_245k"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_FT25Hz_lr1e-4_cyc100k_bs4.5_VQ-RP_CTC-25Hz/checkpoints/step=0245000.ckpt"

model_name="debug"
model_path="/mnt/hdfs/qinxin.025/logs/VQ/25Hz_1x32768_window2.5-2.5sec_stage3/checkpoints/step=0220000.ckpt"
model_cls="recipes.umm2.modules.stages.stage3.Stage3RVQ"
# change the output dir to yours
# out_dir="/mnt/bn/music-llm-nas-lq/qinxin/outputs/tokenizer_eval/"
out_dir=".module_cache/"
nsample=20
slice_modes='even'
slice_dur=60

tasks="loss,locality"
python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample --tasks $tasks --slice_modes $slice_modes --slice_dur $slice_dur

# tasks="ctc_wer"
# python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample --tasks $tasks --slice_modes $slice_modes --slice_dur $slice_dur



# model_name="stage3_FT25Hz_VQ_RP_50Hz_CTC25Hz_1x32768_220k"
# model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage3/50Hz_FT25Hz_lr1e-4_cyc100k_bs4.5_VQ-RP_CTC-25Hz/checkpoints/step=0220000.ckpt"

# tasks="loss,locality"
# python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample --tasks $tasks --slice_modes $slice_modes --slice_dur $slice_dur

# tasks="ctc_wer"
# python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample --tasks $tasks --slice_modes $slice_modes --slice_dur $slice_dur