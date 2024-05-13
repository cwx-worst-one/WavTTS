#!/bin/bash
# Usage: 
#   bash scripts/setup_panther.sh 1.7.14.146 or 
#   OVERRIDE_PANTHER_VERSION=1.7.14.146 bash scripts/setup_panther.sh
#   This script will prioritize using $1.

OVERRIDE_PANTHER_VERSION=${1:-$OVERRIDE_PANTHER_VERSION}
if [ -z "${OVERRIDE_PANTHER_VERSION}" ]
then
    echo "OVERRIDE_PANTHER_VERSION not set, will not update panther"
else
    echo "OVERRIDE_PANTHER_VERSION set, will update panther to ${OVERRIDE_PANTHER_VERSION}"
    PANTHER_TMP=/tmp/panther_${OVERRIDE_PANTHER_VERSION}
    rm -fr ${PANTHER_TMP}
    mkdir -p ${PANTHER_TMP}
    pushd ${PANTHER_TMP}
    wget http://luban-source.byted.org/repository/scm/lab.speech.panther_arnold_${OVERRIDE_PANTHER_VERSION}.tar.gz
    tar -zxf lab.speech.panther_arnold_${OVERRIDE_PANTHER_VERSION}.tar.gz
    sudo pip3 uninstall -y panther-gpu
    pip3 uninstall -y panther-gpu
    sudo -E pip3 install *torch*/panther_gpu-*.whl
    popd
    rm -fr ${PANTHER_TMP}
fi
