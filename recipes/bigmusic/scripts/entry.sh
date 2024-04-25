#!/usr/bin/env bash

export https_proxy=http://bj-rd-proxy.byted.org:3128
export http_proxy=http://bj-rd-proxy.byted.org:3128
export no_proxy=code.byted.org

hdfs dfs -get $1 run.sh
shift

bash run.sh $@
