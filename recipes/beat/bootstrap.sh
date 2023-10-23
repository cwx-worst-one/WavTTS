#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

# Do something before lauching the main program
# e.g. Download some data, install some extra packages, etc.
pip3 install wheel Cython
pip3 install -q -r recipes/beat/requirements.txt
pip3 install numpy==1.23.5

sh launch.sh $@
