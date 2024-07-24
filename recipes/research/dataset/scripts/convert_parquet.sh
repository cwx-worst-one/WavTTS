#!/bin/bash

bash recipes/research/bootstrap.sh exit

pip3 install fastparquet
# python3 recipes/research/dataset/scripts/convert_parquet_mp3.py --rank ${ARNOLD_ID} --world_size ${ARNOLD_WORKER_NUM}
python3 recipes/research/dataset/scripts/convert_parquet_mel.py --rank ${ARNOLD_ID} --world_size ${ARNOLD_WORKER_NUM}