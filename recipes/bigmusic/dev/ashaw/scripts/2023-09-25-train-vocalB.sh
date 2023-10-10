# MCC VocalB dataset study

python3 -m samantha.main fit -c recipes/bigmusic/conf/semantic_modeling/semantic_model_mix_varlen30_07B_us.yaml \
    --run_opts.log_name semantic_model_mulan_text_07B_groupB --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal --run_opts.version baseline

recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/semantic_model_mix_varlen30_07B_us.yaml \
    --run_opts.log_name semantic_model_mulan_text_07B_groupB --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal \
    --run_opts.version groupA_1M --run_opts.region A --run_opts.max_steps 200000 --run_opts.cycle_steps 50000

recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/semantic_model_mix_varlen30_07B_us.yaml \
    --run_opts.log_name semantic_model_mulan_text_07B_groupB --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal \
    --run_opts.version groupB_2M --run_opts.region 2000000 --run_opts.max_steps 200000 --run_opts.cycle_steps 100000

recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/semantic_model_mix_varlen30_07B_us.yaml \
    --run_opts.log_name semantic_model_mulan_text_07B_groupB --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal \
    --run_opts.version groupB_1M --run_opts.region 1000000 --run_opts.max_steps 200000 --run_opts.cycle_steps 50000

recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/semantic_model_mix_varlen30_07B_us.yaml \
    --run_opts.log_name semantic_model_mulan_text_07B_groupB --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal \
    --run_opts.version groupB_500k --run_opts.region 500000 --run_opts.max_steps 200000 --run_opts.cycle_steps 25000

recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/semantic_model_mix_varlen30_07B_us.yaml \
    --run_opts.log_name semantic_model_mulan_text_07B_groupB --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal \
    --run_opts.version groupB_full --run_opts.region B --run_opts.max_steps 200000 --run_opts.cycle_steps 100000