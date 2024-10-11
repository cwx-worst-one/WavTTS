#!/bin/bash -ex

set -x

export NUMEXPR_MAX_THREADS=64
export LD_LIBRARY_PATH=/opt/tiger/jdk/jdk1.8/jre/lib/amd64/server:/opt/tiger/yarn_deploy/hadoop_current/lib/native:/opt/tiger/yarn_deploy/hadoop_current/lib/native/ufs:/opt/tiger/yarn_deploy/hadoop/lib/native:/opt/tiger/yarn_deploy/hadoop_current/lib/native:/opt/tiger/yarn_deploy/hadoop_current/lzo/lib:/usr/local/cuda/lib64:
source scripts/setup_cuda_compat.sh
export PYTHONPATH=/opt/tiger/samantha/mariana:$PYTHONPATH
work_dir=/opt/tiger/samantha
cd $work_dir
export NCCL_DEBUG=WARN
export BYTED_TORCH_C10D_LOG_LEVEL=ERROR

COMMIT=${SAMANTHA_COMMIT:-$(git rev-parse --short HEAD)}

branch_name=$(git branch --show-current)

git fetch origin $COMMIT || echo "fetch $COMMIT failed"
git fetch --unshallow || echo "on a complete repository"

git checkout $COMMIT

git submodule update mariana

bash scripts/data_processing/audio/setup.sh

echo "TORCHRUN scripts/data_processing/audio/feature.py $@"

TORCHRUN scripts/data_processing/audio/feature.py $@

git checkout $branch_name