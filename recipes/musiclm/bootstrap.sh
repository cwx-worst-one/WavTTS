#!/bin/bash -ex

cd $(dirname $0)/../../

pip3 install -qr ./recipes/musiclm/requirements.txt
pip3 install ./recipes/soundstream/torch-museval

sh launch.sh $@