#!/bin/bash
wget "https://www.easyodeaudio.com.cn/api/sail/obj?signature=1f8e41f900e50798d13db48739616700&v=tos%3A7231201152149372965" -O tests/data/44k_3ch.wav
git ls-files tests | grep -e "\.py$" | grep -v unit_test | grep -v batch_test | grep -v dev/data_preprocess.py | grep -v export_test |  xargs python3 -m pytest -m "not disable" --cov-report=xml:coverage.xml --cov=samantha --junit-xml=report.xml
