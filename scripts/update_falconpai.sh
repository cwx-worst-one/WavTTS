#!/bin/sh
#reinstall falconpai
#bash update_falconpai.sh

# Latest version of falconpai.
# This will be updated at dolphin maintainence if needed.
falconpai="1.1.30.0"

case "$1" in
    -h|--help|?)
    echo "Usage: bash update_falconpai.sh arg1"
    echo "       arg1: falconpai_version, default $falconpai"
    echo "If arg1 not provided, latest version will be used."
    exit 0
;;
esac

if [ $# == 1 ]
then
    falconpai=$1
    echo "falconpai: $falconpai"
fi

sudo pip3 uninstall -y byted-falconpai
sudo pip3 uninstall -y byted-falconpai
pip3 uninstall -y byted-falconpai
pip3 uninstall -y byted-falconpai  # double check
if [ "$1" != "" ]; then
  sudo -E pip3 install \
      'http://luban-source.byted.org/repository/scm/lab.speech.falconpai_'$falconpai'.tar.gz'
else
  sudo -E pip3 install -U byted-falconpai \
      --index-url=http://bytedpypi.byted.org/simple \
      --trusted-host=bytedpypi.byted.org
fi
