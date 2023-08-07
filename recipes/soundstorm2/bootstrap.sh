#!/bin/bash -ex

cd $(dirname $0)/../../

pip3 install -qr ./recipes/soundstorm2/requirements.txt
sh launch.sh $@