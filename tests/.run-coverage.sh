#!/bin/bash
hdfs dfs -get /home/byte_speech_sv/user/wangxin.colin/tests/44k_3ch.wav tests/data/44k_3ch.wav

git ls-files tests \
  | grep -e "\.py$" \
  | grep -v benchmarks/dataloader \
  | xargs python3 -m pytest -m "not disable" --cov-report=xml:coverage.xml --cov=samantha --junit-xml=report.xml
