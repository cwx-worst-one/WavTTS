#!/bin/sh

# go to the main dir
THIS_DIR="$( cd "$( dirname "$0" )" && cd "../" && pwd )"
cd $THIS_DIR

if [ "$CUDA_VERSION" == "10" ]; then
    files=$(ls trace.* 2>/dev/null)
else
    files=$(ls trace.*.sqlite 2>/dev/null)
fi

if [ "$files" != "" ]; then
    if [ "$CUDA_VERSION" == "10" ]; then
        python3 -m panther.profiler.nvprof2trace $files
    else
        python3 -m panther.profiler.nsys2trace $files
    fi
fi

for file in $(ls trace.*.json 2>/dev/null)
do
    python3 -m panther.profiler.upload_profile2tos $file
done
