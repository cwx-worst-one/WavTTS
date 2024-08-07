#!/bin/bash
set -x  # for better debug view

source scripts/setup_cuda_compat.sh

CUR_DIR=$(cd $(dirname $0); pwd)
cd $CUR_DIR

# suppress excessive logs
export BYTED_TORCH_C10D_LOG_LEVEL=ERROR

# arnold env: can speed up communication among nodes
export ARNOLD_SORT_IP=1
if [ "$SETUP_MUSIC" == "1" ]
then
    bash setup_music.sh
fi

# install easycycle
pip3 install -q --upgrade pip  bytedance-easycycle==1.1.19 -i https://bytedpypi.byted.org/simple
pip3 install emoji

# setup cruise: install custom cruise version by specify env OVERRIDE_CRUISE_VERSION
if [ -z "$OVERRIDE_CRUISE_VERSION" ]
then
    echo "OVERRIDE_CRUISE_VERSION not set, will not update cruise"
else
    bash scripts/setup_cruise.sh $OVERRIDE_CRUISE_VERSION
    # cruise will be installed into /opt/tiger/cruise, set PYTHONPATH to make it valid.
    export PYTHONPATH=/opt/tiger/cruise:$PYTHONPATH
fi

if [ -z "${OVERRIDE_FMHA_PLUS_VERSION}" ]
then
    echo "OVERRIDE_FMHA_PLUS_VERSION not set, will not update fmha_plus"
else
    bash scripts/setup_fmha_plus.sh $OVERRIDE_FMHA_PLUS_VERSION
fi

if [ -z "${OVERRIDE_PANTHER_VERSION}" ]
then
    echo "OVERRIDE_PANTHER_VERSION not set, will not update panther"
else
    bash scripts/setup_panther.sh $OVERRIDE_PANTHER_VERSION
fi

export MASTER_PORT=${METIS_WORKER_0_PORT}
export MASTER_ADDR=${METIS_WORKER_0_HOST}
export NODE_RANK=${ARNOLD_ID}
export NNODES=${ARNOLD_WORKER_NUM}
export NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
export ARNOLD_OUTPUT=${ARNOLD_OUTPUT}

echo "TOTAL WORKERS    :   ${NNODES}"
echo "CURRENT WORKER ID:   ${NODE_RANK}"
echo "#GPU PER-WORKER  :   ${NPROC_PER_NODE}"
echo "MASTER NODE IP   :   ${MASTER_ADDR}"
echo "MASTER NODE PORT :   ${MASTER_PORT}"
echo "WORLD SIZE       :   ${WORLD_SIZE}"
echo "ARNOLD OUTPUT    :   ${ARNOLD_OUTPUT}"

export OMP_NUM_THREADS=8

if [ "${ARNOLD_DEVICE_TYPE#*A100*}" != "$ARNOLD_DEVICE_TYPE" ]; then
  IB_HCA=mlx5
elif [ "${ARNOLD_DEVICE_TYPE#*H800*}" != "$ARNOLD_DEVICE_TYPE" ]; then
  IB_HCA=mlx5
else
  IB_HCA=$ARNOLD_RDMA_DEVICE:1
fi

if [ "$ARNOLD_RDMA_DEVICE" != "" ]; then
   export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:=0}
   export NCCL_IB_HCA=${NCCL_IB_HCA:=$IB_HCA}
   export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
   export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:=eth0}
else
   export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:=1}
   export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:=eth0}
fi

echo "IB_HCA    :   ${IB_HCA}"
echo "NCCL_IB_DISABLE    :   ${NCCL_IB_DISABLE}"

export NCCL_DEBUG=${NCCL_DEBUG:=WARN}

# initiate actions using main.py
# check if TORCHRUN is available
if [ -x "$(command -v TORCHRUN)" ]; then
    CMD="TORCHRUN"
else
    CMD="./TORCHRUN"
fi

if [ $NODE_RANK -eq 0 ]; then
  trap "python3 scripts/data_processing/bigtts/feature_model_callback.py" EXIT
fi

echo "Use launcher: ${CMD}"

$CMD -m samantha.main $@

ret=$?
echo "Samantha exit time: $(date)"
exit $ret
