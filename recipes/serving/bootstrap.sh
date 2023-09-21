#!/usr/bin/env bash
pip3 install --upgrade pip
pip3 install -r recipes/serving/requirements.txt

# start the euler server
worker_num=${APP_WORKER_NUM:-1}
python3 -m bytedunicorn -k euler.worker.SyncWorker -w ${worker_num} -b [::]:8888 -t 3600 recipes.serving.server:server
