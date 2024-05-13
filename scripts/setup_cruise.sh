#!/bin/bash
# Setup cruise: install custom cruise version by specify env OVERRIDE_CRUISE_VERSION
# Usage:
#   While cruise version is 1.0.0.x, run these commands below as you like
#   bash scripts/setup_cruise.sh x or OVERRIDE_PANTHER_VERSION=x bash scripts/setup_cruise.sh
#   This script will prioritize using $1.
# Note:
#   Cruise will be installed into /opt/tiger/cruise, don't forget your PYTHONPATH.

OVERRIDE_CRUISE_VERSION=${1:-OVERRIDE_CRUISE_VERSION}
if [ -z "$OVERRIDE_CRUISE_VERSION" ]
then
    echo "OVERRIDE_CRUISE_VERSION not set, will not update cruise"
else
    echo "OVERRIDE_CRUISE_VERSION set, will update cruise to 1.0.0.$OVERRIDE_CRUISE_VERSION"
    pushd /opt/tiger
    rm -rf cruise;
    mkdir -p cruise && cd cruise;
    wget http://luban-source.byted.org/repository/scm/data.aml.cruise_1.0.0.$OVERRIDE_CRUISE_VERSION.tar.gz;
    tar -xf data.aml.cruise*.tar.gz;
    popd
fi
