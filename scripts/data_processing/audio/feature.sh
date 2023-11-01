#!/bin/bash -ex

pip3 install resemblyzer==0.1.4 -i https://bytedpypi.byted.org/simple

export LD_LIBRARY_PATH=/opt/tiger/jdk/jdk1.8/jre/lib/amd64/server:/opt/tiger/yarn_deploy/hadoop_current/lib/native:/opt/tiger/yarn_deploy/hadoop_current/lib/native/ufs:/opt/tiger/yarn_deploy/hadoop/lib/native:/opt/tiger/yarn_deploy/hadoop_current/lib/native:/opt/tiger/yarn_deploy/hadoop_current/lzo/lib:/usr/local/cuda/lib64::/usr/local/cuda/compat

work_dir=/opt/tiger/samantha
cd $work_dir
export NCCL_DEBUG=WARN
export BYTED_TORCH_C10D_LOG_LEVEL=ERROR

COMMIT=${SAMANTHA_COMMIT:-$(git rev-parse --short HEAD)}

git fetch --all

git checkout $COMMIT

echo "TORCHRUN scripts/data_processing/audio/feature.py $@"

TORCHRUN scripts/data_processing/audio/feature.py $@