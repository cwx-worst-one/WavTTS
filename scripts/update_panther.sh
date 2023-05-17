#!/bin/sh
#install new version panther
#bash update_panther.sh 1.0.0.5 1.6.0.2
SCM_URL_PREFIX='http://luban-source.byted.org/repository/scm'
function version_le() { test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" == "$1"; }

# Latest version of panther.
# This will be updated at dolphin maintainence if needed.
panther_arnold=1.7.5.0

case "$1" in
    -h|--help|?)
    echo "Usage:bash update_panther.sh arg1 arg2"
    echo "      arg1: tensorflow_onnx_panther_version"
    echo "      arg2: panther_arnold_version $panther_arnold"
    echo "If arg2 not available, latest version will be used"
    exit 0
;;
esac

if [ "$1" != "" ]; then
  tensorflow_onnx_panther=$1
    tf2onnx_version=1.9.3
  if version_le "${tensorflow_onnx_panther}" "1.0.0.17"; then
    tf2onnx_version=1.9.2
  fi
  cd /tmp
  rm -f \
      "lab.speech.tensorflow_onnx_panther_${tensorflow_onnx_panther}.tar.gz" \
      tf2onnx-${tf2onnx_version}-py3-none-any.whl
  wget "${SCM_URL_PREFIX}/lab.speech.tensorflow_onnx_panther_${tensorflow_onnx_panther}.tar.gz"
  tar -xvf "lab.speech.tensorflow_onnx_panther_${tensorflow_onnx_panther}.tar.gz"
  sudo pip3 uninstall -y tf2onnx
  sudo pip3 uninstall -y tf2onnx
  pip3 uninstall -y tf2onnx
  pip3 uninstall -y tf2onnx  # double check
  sudo -E pip3 install \
      tf2onnx-${tf2onnx_version}-py3-none-any.whl \
      --trusted-host bytedpypi.byted.org
fi

if [ "$2" != "" ]; then
  panther_arnold=$2
  echo "panther: $panther_arnold"
fi

panther_version=`echo ${panther_arnold} | awk -F "." '{print $1"."$2"."$3}'`

cd /tmp
rm -rf \
    "lab.speech.panther_arnold_${panther_arnold}.tar.gz" \
    *torch*
wget "${SCM_URL_PREFIX}/lab.speech.panther_arnold_${panther_arnold}.tar.gz"
tar -xvf "lab.speech.panther_arnold_${panther_arnold}.tar.gz"
sudo pip3 uninstall -y panther-gpu
sudo pip3 uninstall -y panther-gpu
pip3 uninstall -y panther-gpu
pip3 uninstall -y panther-gpu # double check
sudo -E pip3 install \
    *torch*/panther_gpu-*-linux_x86_64.whl
