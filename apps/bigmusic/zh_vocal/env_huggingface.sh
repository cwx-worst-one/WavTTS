cd /opt/tiger/samantha

mkdir -p .module_cache/huggingface/
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bochen/model_cache/larger_clap_general .module_cache/huggingface/
# hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bochen/model_cache/opus-mt-zh-en .module_cache/huggingface/
