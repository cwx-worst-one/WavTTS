#!/bin/bash
set -e
set -x

# disable logging config to make cruise logging work
export SAMANTHA_DISABLE_LOGGING_BASIC_CONFIG=1
MAX_JOBS=${MAX_JOBS:-8}

ulimit -l unlimited # CI machine not config this

# run unit tests in folder `apps/bigmusic/mariana_tasks/tests`
pytest -m "not disable" -n $MAX_JOBS apps/bigmusic/mariana_tasks/tests -sx