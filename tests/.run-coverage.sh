#!/bin/bash
# if not exist wav file, download it from hdfs
if [ ! -f tests/data/44k_3ch.wav ]; then
  echo "Downloading wav file from hdfs"
  hdfs dfs -get /home/byte_speech_sv/user/wangxin.colin/tests/44k_3ch.wav tests/data/
fi

# install cruise
mkdir -p /opt/tiger/cruise && cd /opt/tiger/cruise;
wget http://luban-source.byted.org/repository/scm/data.aml.cruise_1.0.0.2277.tar.gz;
tar -xf data.aml.cruise*.tar.gz;
export PYTHONPATH=/opt/tiger/cruise:$PYTHONPATH
cd -

git ls-files tests \
  | grep -e "\.py$" \
  | grep -v benchmarks/dataloader \
  | xargs python3 -m pytest -m "not disable" --cov-report=xml:coverage.xml --cov=samantha --junit-xml=report.xml
