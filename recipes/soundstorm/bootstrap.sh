#!/bin/bash -ex

cd $(dirname $0)/../../

pip3 install -qr ./recipes/soundstorm/requirements.txt

sh launch.sh $@