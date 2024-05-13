#!/bin/bash
# Usage: 
#   bash scripts/setup_fmha_plus 1.7.14.146 or 
#   OVERRIDE_FMHA_PLUS_VERSION=1.7.14.146 bash scripts/setup_fmha_plus.sh
#   This script will prioritize using $1.

OVERRIDE_FMHA_PLUS_VERSION=${1:-$OVERRIDE_FMHA_PLUS_VERSION}
if [ -z "${OVERRIDE_FMHA_PLUS_VERSION}" ]
then
    echo "OVERRIDE_FMHA_PLUS_VERSION not set, will not update fmha_plus"
else
    echo "OVERRIDE_FMHA_PLUS_VERSION set, will update fmha_plus to ${OVERRIDE_FMHA_PLUS_VERSION}"
    # at least >= 1.0.0.7
    FMHA_TMP=/tmp/fmha_${OVERRIDE_FMHA_PLUS_VERSION}
    mkdir -p ${FMHA_TMP}
    pushd ${FMHA_TMP}
    wget http://luban-source.byted.org/repository/scm/data.speech.flash_attn_plus_${OVERRIDE_FMHA_PLUS_VERSION}.tar.gz
    sudo pip3 uninstall -y fmha_plus
    pip3 uninstall -y fmha_plus
    tar -zxf data.speech.flash_attn_plus_${OVERRIDE_FMHA_PLUS_VERSION}.tar.gz
    sudo -E pip3 install --no-deps fmha_plus-*.whl
    popd
    rm -fr ${FMHA_TMP}
fi
