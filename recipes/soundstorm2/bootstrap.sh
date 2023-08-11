#!/bin/bash -ex

cd $(dirname $0)/../../

pip3 install -qr ./recipes/soundstorm2/requirements.txt

# DAC
pip3 install descript-audio-codec descript-audiotools --no-deps
pip3 install -q flatten_dict randomname argbind

mkdir -p /home/tiger/.cache/descript/dac
hdfs dfs -get hdfs://haruna/home/byte_speech_sv/models/dac/weights_44khz_8kbps_0.0.1.pth /home/tiger/.cache/descript/dac
# Only for CN HDFS:
# export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
# export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
# mkdir -p /home/tiger/.cache/descript/dac
# hdfs dfs -get hdfs://haruna/home/byte_speech_sv/models/dac/weights_44khz_8kbps_0.0.1.pth /home/tiger/.cache/descript/dac

sh launch.sh $@