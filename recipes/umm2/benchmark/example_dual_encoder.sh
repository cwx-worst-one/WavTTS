model_name="test"

model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage4/25Hz_fixR0_RVQ-RP4x32768_GAN/checkpoints/step=0205000.ckpt"
model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/stage4/25Hz_fixR0_concat_RVQ-RP4x32768_GAN_v2/checkpoints/step=0260000.ckpt"
model_cls="recipes.umm2.modules.stages.dual_encoder.DualEncoder"
out_dir="./.module_cache/"
nsample=100

python3 recipes/umm2/benchmark/run_eval_dual_encoder.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample
# tasks="loss,locality"
# python3 recipes/umm2/benchmark/run_eval.py --model_name $model_name --model_cls $model_cls --model_path $model_path --out_dir $out_dir --nsample $nsample --tasks $tasks
