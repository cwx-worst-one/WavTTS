#!/bin/bash -ex

cd $(dirname $0)/../../
echo "Working DIR: $(pwd)"

hdfs dfs -get hdfs://haruna/home/byte_speech_sv/zongyu.yin/assets/mcc60m_index_train.txt recipes/datasets/mcc/mcc60m_index_train.txt
hdfs dfs -get hdfs://haruna/home/byte_speech_sv/zongyu.yin/assets/mcc60m_index_val.txt recipes/datasets/mcc/mcc60m_index_val.txt

bash launch.sh $@