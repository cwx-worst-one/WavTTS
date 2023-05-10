#!/bin/sh

export ARNOLD_PROFILER=2
export PROFILER_CMD="nvprof -f -o trace.%p --profile-from-start off"

# do run
if [ ! -f /opt/tiger/arnold/arnold_entrypoint/tools/MPIRUN ]; then
    ${PROFILER_CMD} python3 train.py --train.profile_start_step 100 --train.profile_end_step 200 "$@"
else
    MPIRUN python3 train.py "$@"
fi


files=$(ls trace.* 2>/dev/null)
if [ "$files" != "" ]; then
    pip3 install cxxfilt --trusted-host=bytedpypi.byted.org
    python3 scripts/nccl_analysis.py $files
    container_id=$(cat /proc/self/cpuset | awk -F / '{print $4}')
    stderr_file=$(ls /data00/yarn/userlogs/*/${container_id}/stderr)
    cat *.json > $stderr_file
fi
