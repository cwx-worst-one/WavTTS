#!/bin/bash

CURR_PATH=$(cd $(dirname $0); pwd)

hdfs dfs -get /home/byte_speech_sv/jingsong.gao/sami_ai_llm/data/dev-clean_mhubert-km1000.jsonl $CURR_PATH
