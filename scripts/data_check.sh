#!/bin/sh
#data or path check
#bash data_check.sh 
set -x  # for better debug view

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR

export FALCONPAI_DISABLE_TF=1

python3 data_check.py "$@"
exit 0