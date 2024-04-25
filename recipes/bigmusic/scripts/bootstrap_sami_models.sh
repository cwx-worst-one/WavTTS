#!/bin/bash

# Install MIR models

BIGMUSIC_SAMI_MODELS_DIR="/opt/tiger/bigmusic_sami_models"

git clone -b yilin/serve --single-branch git@code.byted.org:seed/bigmusic_sami_models.git $BIGMUSIC_SAMI_MODELS_DIR

cd $BIGMUSIC_SAMI_MODELS_DIR
pip3 install -e .
pip3 install protobuf==3.20
cd -