#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

pip3 install -q --upgrade pip -i https://bytedpypi.byted.org/simple
pip3 install -q -r recipes/mulan/requirements.txt
pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple
pip3 install -i https://bytedpypi.byted.org/simple http://luban-source.byted.org/repository/scm/data.aml.cruise_1.0.0.1260.tar.gz
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

export LD_LIBRARY_PATH=/opt/tiger/native_libhdfs/lib/native:$LD_LIBRARY_PATH
export ARNOLD_HDFS_NATIVE=1
export ARNOLD_HDFS_CELER=1
export INFSEC_HADOOP_ENABLED=1
export CPP_HDFS_CONF=/opt/tiger/arnold/hdfs_client/conf/celer_us/core-site.xml:/opt/tiger/arnold/hdfs_client/conf/celer_us/hdfs-site.xml

