#!/bin/bash -ex

# suppress excessive logs
export BYTED_TORCH_C10D_LOG_LEVEL=ERROR


CUR_DIR=$(cd $(dirname $0); pwd)
cd $CUR_DIR

export MASTER_PORT=${METIS_WORKER_0_PORT}
export MASTER_ADDR=${METIS_WORKER_0_HOST}
export NODE_RANK=${ARNOLD_ID}
export NNODES=${ARNOLD_WORKER_NUM}
export NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
export ARNOLD_OUTPUT=${ARNOLD_OUTPUT}
export SAIL_EXPERIMENT_ID=${SAIL_EXPERIMENT_ID}

echo "TOTAL WORKERS    :   ${NNODES}"
echo "CURRENT WORKER ID:   ${NODE_RANK}"
echo "#GPU PER-WORKER  :   ${NPROC_PER_NODE}"
echo "MASTER NODE IP   :   ${MASTER_ADDR}"
echo "MASTER NODE PORT :   ${MASTER_PORT}"
echo "WORLD SIZE       :   ${WORLD_SIZE}"
echo "ARNOLD OUTPUT    :   ${ARNOLD_OUTPUT}"
echo "SAIL EXP ID      :   ${SAIL_EXPERIMENT_ID}"

# export BYTED_TORCH_BYTECCL=O3 # enable byteps

export OMP_NUM_THREADS=8

# set up nccl relevant env, for more detail,
# please refer https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

# patch triton
sudo cp samantha/utils/patches/matmul.py /usr/local/lib/python3.9/dist-packages/triton/ops/blocksparse/ || echo
sudo cp samantha/utils/patches/matmul.py /home/tiger/.local/lib/python3.9/site-packages/triton/ops/blocksparse/ || echo

# initiate actions using main.py
# check if TORCHRUN is available
if [ -x "$(command -v TORCHRUN)" ]; then
    CMD="TORCHRUN"
else
    CMD="python3"
fi

echo "Use launcher: ${CMD}"

$CMD -m samantha.main $@
