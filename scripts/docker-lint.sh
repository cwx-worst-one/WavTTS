#!/bin/bash

which docker
docker=$?
if [ $docker == 0 ]
then
    #run pylint in docker
    echo "run lint in docker"
    DOLPHIN_ROOT=$(cd "$(dirname "$0")/..";pwd)
    echo "DOLPHIN_ROOT:$DOLPHIN_ROOT"
    mkdir -p $HOME/.dolphin_local
    chmod -R 777 $HOME/.dolphin_local
    docker run \
        --user $(id -u):$(id -g) \
        --rm \
        -v $HOME/.dolphin_local:/.local \
        -v $DOLPHIN_ROOT:/dolphin \
        --workdir=/dolphin \
        hub.byted.org/arnold/dolphin:torch1.8.1_cuda11.1_dolphin \
        bash -c "python3 -m black -l 100 -S --safe core/ tests/ train.py; python3 -m pylint -j 16 --rcfile=.pylintrc train.py core/ tests/"
else
    #run pylint in local
    echo "run lint in local"
    pip3 install pylint==2.15.4 black==22.10.0
    python3 -m black -l 100 -S --safe core/ tests/ train.py
    python3 -m pylint -j 16 --rcfile=.pylintrc train.py core/ tests/
fi
