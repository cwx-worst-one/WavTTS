#!/bin/bash

if [ $ARNOLD_PROFILER ] && [ "$ARNOLD_PROFILER" -gt 0 ]; then
  echo "enable dolphin profiler"
  # setup cuda version
  /usr/local/cuda/bin/nvcc --version | grep release\ 11.*
  cuda_10=$?
  if [ "$cuda_10" == "1" ]; then
    export CUDA_VERSION='10'
  else
    export CUDA_VERSION='11'
  fi

  python3 -m panther.profiler.setup_env

  if [ "$CUDA_VERSION" == "10" ]; then
    # using nvprof profiler
    export PROFILER_CMD="nvprof -f -o trace.%p --profile-from-start off"
  else
    # using nsys profiler
    export PROFILER_CMD="nsys profile -o trace.%p --export sqlite -d 300 --kill none -c cudaProfilerApi"
  fi

  pip3 install cxxfilt
  rm -f trace.*
fi
