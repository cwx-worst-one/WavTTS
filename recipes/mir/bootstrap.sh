#!/bin/bash

# add proxy for CN region
if [ "$ARNOLD_REGION" == "CN" ]; then
	echo "Adding network proxy"
  export https_proxy=http://sys-proxy-rd-relay.byted.org:8118 http_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy="byted.org"
fi

pip3 install -qr ./recipes/bigmusic/requirements.txt
bash launch.sh $@