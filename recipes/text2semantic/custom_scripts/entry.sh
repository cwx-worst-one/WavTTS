#!/usr/bin/env bash

export https_proxy=http://bj-rd-proxy.byted.org:3128
export http_proxy=http://bj-rd-proxy.byted.org:3128
export no_proxy=code.byted.org

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd ${THIS_DIR}

hdfs dfs -get $1 ./run.sh
shift
bash run.sh $@
