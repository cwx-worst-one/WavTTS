#! /bin/bash

reuse=(
    original
	io_reuse
    item_reuse
    batch_reuse
)
failed=0
for config in ${reuse[@]}; do
    echo $config
    rm -rf "/tmp/falconreader"
    rm -rf "/dev/shm/falconreader_*"
    mpirun -np $ARNOLD_WORKER_GPU python3 train.py $@ --data.$config 10 --train.max_iters 2000 --train.remote_save_root 0
    exit_code=$?
    if [ $exit_code != 0 ]; then
        failed=1
        continue
    fi
done

export DOLPHIN_DATA_TRANSFORM_PROFILE=1
rm -rf "/tmp/falconreader"
rm -rf "/dev/shm/falconreader_*"
mpirun -np $ARNOLD_WORKER_GPU python3 train.py $@ --train.max_iters 200 --data.prefetch_worker_num 1 --train.remote_save_root 0
exit_code=$?
if [ $exit_code != 0 ]; then
    failed=1
fi
if [ $failed == 1 ]; then
    exit 1
fi
