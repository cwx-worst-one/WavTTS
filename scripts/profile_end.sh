#!/bin/bash

if [ $ARNOLD_PROFILER ] && [ "$ARNOLD_PROFILER" -gt 0 ]; then
  # NOTE(zhengyijie): Prevent segment faults when converting trace to json
  echo "Training is complete, profile result conversion is in progress"
  bash -x scripts/trace2json.sh
fi