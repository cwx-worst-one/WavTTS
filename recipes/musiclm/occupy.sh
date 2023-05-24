#!/bin/bash

cd $(dirname $0)/../../

pip3 install -qr ./recipes/musiclm/requirements.txt
python3 ./recipes/musiclm/scripts/occupy.py $@