#!/bin/bash -ex

set -x

# python3 nastk.py cp -s $1 -t ./
cp $1 ./
local_shell=`pwd`/$(basename $1)

bash $local_shell ${@:2}
