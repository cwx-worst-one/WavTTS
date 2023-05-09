#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

# Do something before lauching the main program
# e.g. Download some data, install some extra packages, etc.

sh launch.sh $@
