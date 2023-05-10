#!/bin/bash
set -e
set -x
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"

INPUT_DIR=$1
OUTPUT_DIR=$2
TEMPLATE=$3
VERSION=$4
COPY_DIR=$5

if [ -d "petrel_asr_script" ]; then
  rm -r petrel_asr_script
fi
mkdir petrel_asr_script

if [ "$VERSION" = "online" ]; then
  url="https://luban-source.byted.org/repository/scm/api/v1/download_latest/?name=lab/speech/petrel_asr_script&type=online"
elif [ "$VERSION" = "test" ]; then
  url="https://luban-source.byted.org/repository/scm/api/v1/download_latest/?name=lab/speech/petrel_asr_script&type=test"
else
  url="http://luban-source.byted.org/repository/scm/lab.speech.petrel_asr_script_$VERSION.tar.gz"
fi

wget $url -O "lab.speech.petrel_asr_script.tar.gz"
tar zxf lab.speech.petrel_asr_script.tar.gz -C petrel_asr_script


if [ -d "$OUTPUT_DIR" ]; then
  rm -r "$OUTPUT_DIR"
fi

if [ ! -e "petrel_asr_script/template/${TEMPLATE}.ini" ]; then
  echo "cannot find template for ${TEMPLATE}"
  exit 1
fi

python3 petrel_asr_script/package.py \
  --resource_dir=${INPUT_DIR} \
  --penguin_config=${INPUT_DIR}/config.ini \
  --dst_dir=${OUTPUT_DIR} \
  --engine_preset=petrel_asr_script/template/${TEMPLATE}.ini \
  --copy_resource_path=$COPY_DIR
