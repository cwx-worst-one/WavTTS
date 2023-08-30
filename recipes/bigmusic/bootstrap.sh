#!/bin/bash -ex

cd $(dirname $0)/../../

sudo apt update
sudo apt install espeak -y
sudo apt install ffmpeg -y
pip3 install -qr ./recipes/bigmusic/requirements.txt

sh launch.sh $@