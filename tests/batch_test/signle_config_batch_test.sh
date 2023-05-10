#! /bin/bash
# for reproducity of LSTM module.
export CUBLAS_WORKSPACE_CONFIG=:16:8

export PYTHONPATH=.:
python3 tests/batch_test/asr_batch_test.py "$@"