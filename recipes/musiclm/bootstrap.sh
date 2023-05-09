#!/bin/bash -ex

cd $(dirname $0)/../../

pip3 install -qr ./recipes/musiclm/requirements.txt

sh launch.sh $@