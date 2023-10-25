#!/usr/bin/env bash
pip3 install --upgrade pip
pip3 install -r recipes/serving/requirements.txt

hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource ./resource

# start the euler server
worker_num=${APP_WORKER_NUM:-1}
python3 -m bytedunicorn -k euler.worker.SyncWorker -w ${worker_num} -b [::]:8888 -t 3600 recipes.serving.server:server
