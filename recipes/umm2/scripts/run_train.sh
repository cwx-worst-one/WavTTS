sh /opt/tiger/samantha/launch.sh fit \
    --config recipes/umm2/conf/umm_stage3_reg_2255mixedZHEN_2375Speech.yaml \
    --run_opts.version debug \
    --run_opts.log_name debug \
    --run_opts.hdfs_log_dir "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs" \
    --run_opts.precision "16-mixed"