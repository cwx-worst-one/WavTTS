#!/bin/bash
set -x  # for better debug view

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR

export FALCONPAI_DISABLE_TF=1

# Sort arnold ip to speed up nccl communication speed,
# see: https://bytedance.feishu.cn/docx/Ohrrdue6OoKtyhxeKeBc6641nNe
export ARNOLD_NETWORK_TOPOLOGY="SAME_NETMINIPOD"
export ARNOLD_SORT_IP=1


# echo dolphin scm version
current_revision_path="current_revision"
if [ -f "$current_revision_path" ]; then
  cat "$current_revision_path"
fi

if [ "$ARNOLD_ROLE" == "scheduler" ]; then
  export DMLC_ENABLE_RDMA=${DMLC_ENABLE_RDMA:-1}
  DMLC_NUM_SERVER=$ARNOLD_WORKER_NUM \
  DMLC_NUM_WORKER=$ARNOLD_WORKER_NUM \
  DMLC_ROLE=scheduler \
  bpslaunch
  exit 0
fi
# if global shuffle, some meta data will be write to the file
# so we need clean it before training
rm -rf "/tmp/falconreader"
rm -rf "/dev/shm/falconreader_*"

# Arnold scheduler does not need to setup the following
if [ "$ARNOLD_WORKER_NUM" == "1" ]; then
  DIST_INIT_FILE_PATH="dist_init_file${ARNOLD_TRIAL_ID}"
  rm -rf $DIST_INIT_FILE_PATH
  touch $DIST_INIT_FILE_PATH
fi

if [[ "$@" == *debug-arnold-cpu* ]]; then
  echo "debug arnold"
  sleep infinity
fi

# set pip3 source in CN, no need in US.
if [ "$ARNOLD_REGION" == "CN" ]; then
  pip3 config set install.trusted-host bytedpypi.byted.org
fi
if [ "x${PANTHER_ARNOLD_VERSION}" != "x" ]; then
  bash -x scripts/update_panther.sh "" ${PANTHER_ARNOLD_VERSION}
fi
if [ "x${FALCONPAI_VERSION}" != "x" ]; then
  bash -x scripts/update_falconpai.sh ${FALCONPAI_VERSION}
fi
if [ "x${ASR_EVAL_TOOL_VERSION}" != "x" ]; then
  bash -x scripts/update_asr_eval_tool.sh ${ASR_EVAL_TOOL_VERSION} ${I18N_TEXT_FORMAT_VERSION}
fi

if [ -d "/opt/tiger/byteslim" ]; then
  pip3 install /opt/tiger/byteslim/*.whl -i https://bytedpypi.byted.org/simple
  pip3 show byteslim
fi



# setup profile config
source scripts/profile_start.sh

# debug core dump
if [ "${DOLPHIN_DEBUG_CORE}" == "1" ]; then
  rm -rf ./gdb.*
  echo "set pagination off
  set logging file ./gdb.log
  set logging on
  set print elements 30000
  r
  thread apply all bt
  q
  y
  " > cmds
  export DOLPHIN_CMD_PREFIX="gdb -x cmds --args"
fi

# analyze IO bottleneck
if [ "$DATA_TIME_PROFILE" == "1" ]; then
  unset ARNOLD_PROFILER
  bash -x scripts/data_time.sh "$@"
  export ARNOLD_PROFILER=2
fi
mpirun -np $ARNOLD_WORKER_GPU ${DOLPHIN_CMD_PREFIX} ${PROFILER_CMD} python3 train.py "$@"
exit_code=$?
unset LD_PRELOAD

# upload Arnold local logs
PYTHONPATH=$PYTHONPATH:$(pwd) python3 scripts/upload_log.py "$@"

# output profile result
source scripts/profile_end.sh

# wait for all child processes to finish executing, especially HDFS related processes
sleep 30
rm -rf "/tmp/falconreader"
rm -rf "/dev/shm/falconreader_*"
exit $exit_code
