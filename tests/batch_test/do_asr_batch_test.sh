#! /bin/bash
# for reproducity of LSTM module.
export CUBLAS_WORKSPACE_CONFIG=:16:8

# start batch test in multi progress to
# avoid the influence between different trainings
export PYTHONPATH=.:

configs=(
    configs/asr/child_zh_time_reduce_online_prosody_baseline.py
    configs/asr/dfsmn_rnnt.py
    configs/asr/librispeech_rnnt_lstmp.py
    configs/asr/librispeech_rnnt_hmm_free.py
    configs/asr/librispeech_rnnt.py
    configs/asr/tel_rnnt_transfomer_80kh.py
)
failed=0
for config in ${configs[@]}; do
    python3 tests/batch_test/asr_batch_test.py --config $config
    exit_code=$?
    if [ $exit_code != 0 ]; then
        failed=1
        continue
    fi
done
if [ $failed == 1 ]; then
    exit 1
fi