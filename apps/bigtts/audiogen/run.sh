mkdir .deploy_cache
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/zhuangxiaobin/v2a_deploy/models--apple--DFN2B-CLIP-ViT-B-16 .deploy_cache
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/zhuangxiaobin/v2a_deploy/v2a_0.7b_0.3_v2_20k_encoder.ckpt .deploy_cache
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/zhuangxiaobin/v2a_deploy/v2a_0.7b_0.3_v2_20k_diffusion.ckpt .deploy_cache
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/zhuangxiaobin/v2a_deploy/v2a_0.7b_0.3_v2_20k_vocoder.ckpt .deploy_cache
# python3 apps/bigtts/audiogen/soundify/v2a_infer.py 
