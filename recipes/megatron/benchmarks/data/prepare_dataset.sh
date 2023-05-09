#!/bin/bash

python3 -c 'from datasets import load_dataset; ds = load_dataset("openwebtext", split="train", keep_in_memory=False); ds.to_json("openwebtext.jsonl", orient="records", lines=True, force_ascii=False)'

hdfs dfs -mkdir -p /home/byte_speech_sv/jingsong.gao/sami_ai_llm/data/
hdfs dfs -put openwebtext.jsonl /home/byte_speech_sv/jingsong.gao/sami_ai_llm/data/
rm openwebtext.jsonl
