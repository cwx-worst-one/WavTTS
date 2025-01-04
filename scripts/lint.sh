#!/bin/bash -e
pip3 install --quiet pre-commit==4.0.1
export http_proxy="http://sys-proxy-rd-relay.byted.org:8118" \
    https_proxy="http://sys-proxy-rd-relay.byted.org:8118" \
    no_proxy="byted.org,anaconda.org"
pre-commit run --all-files