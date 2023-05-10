#!/bin/sh
#reinstall dataloader
#bash update_dataloader.sh

# Latest version of dataloader.
# This will be updated at dolphin maintainence if needed.
dataloader="0.3.6"

case "$1" in
    -h|--help|?)
    echo "Usage: bash update_dataloader.sh arg1"
    echo "       arg1: dataloader_version, default $dataloader"
    echo "If arg1 not provided, latest version will be used."
    exit 0
;;
esac

if [ $# == 1 ]
then
    dataloader=$1
    echo "dataloader: $dataloader"
fi

sudo pip3 uninstall -y byted-dataloader
sudo pip3 uninstall -y byted-dataloader
pip3 uninstall -y byted-dataloader
pip3 uninstall -y byted-dataloader  # double check
if [ "$1" != "" ]; then
  sudo -E pip3 install byted-dataloader==$dataloader -i https://bytedpypi.byted.org/simple
else
  sudo -E pip3 install -U byted-dataloader \
      --index-url=http://bytedpypi.byted.org/simple \
      --trusted-host=bytedpypi.byted.org
fi
