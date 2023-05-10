#!/bin/sh
#reinstall falconclaw
#bash update_falconclaw.sh 1.0.5.72

# Latest version of falconclaw.
# This will be updated at dolphin maintainence if needed.
falconclaw='1.0.5.72'

case "$1" in
    -h|--help|?)
    echo "Usage: bash update_falconclaw.sh arg1"
    echo "       arg1: falconclaw_version, default $falconclaw"
    echo "If arg1 not provided, latest version will be used."
    exit 0
;;
esac

if [ $# == 1 ]
then
    falconclaw=$1
    echo "falconclaw: $falconclaw"
fi
pip3 install --force-reinstall 'http://luban-source.byted.org/repository/scm/lab.speech.falconclaw_'$falconclaw'.tar.gz'
