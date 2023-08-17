#!/bin/bash -ex


# pip 
sudo pip3 install -U bytedance.easycycle==0.0.1.post29

# suppress excessive logs
export BYTED_TORCH_C10D_LOG_LEVEL=ERROR

# setting hdfs envs
export LD_LIBRARY_PATH=/opt/tiger/native_libhdfs/lib/native:$LD_LIBRARY_PATH
export ARNOLD_HDFS_NATIVE=1
export ARNOLD_HDFS_CELER=1
export INFSEC_HADOOP_ENABLED=1
export CPP_HDFS_CONF=/opt/tiger/arnold/hdfs_client/conf/celer_us/core-site.xml:/opt/tiger/arnold/hdfs_client/conf/celer_us/hdfs-site.xml

# arnold env: can speed up communication among nodes
export ARNOLD_SORT_IP=1

# setup cruise: install custom cruise version by specify env OVERRIDE_CRUISE_VERSION
if [ -z "$OVERRIDE_CRUISE_VERSION" ]
then
    echo "OVERRIDE_CRUISE_VERSION not set, will not update cruise"
else
    echo "OVERRIDE_CRUISE_VERSION set, will update cruise to $OVERRIDE_CRUISE_VERSION"
    cd /opt/tiger;
    rm -rf cruise;
    mkdir -p cruise && cd cruise;
    wget http://luban-source.byted.org/repository/scm/data.aml.cruise_1.0.0.$OVERRIDE_CRUISE_VERSION.tar.gz;
    tar -xf data.aml.cruise*.tar.gz;
    export PYTHONPATH=/opt/tiger/cruise:$PYTHONPATH
fi


CUR_DIR=$(cd $(dirname $0); pwd)
cd $CUR_DIR

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
