#!/bin/bash
git clone https://github.com/k2-fsa/libriheavy

cd libriheavy
bash run.sh --stage 1 --stop-stage 1
bash run.sh --stage 2 --stop-stage 2