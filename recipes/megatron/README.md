# Megatron Tasks

This folder is for developing megatron based training tasks.

We include `Megatron-LM` as git submodule under `llm` folder. To fetch the content, run:

`git submodule update --init`

We provide a unified entry script `recipes/megatron/bootstrap.sh` to install dependencies and setup environment for Megatron. It's recommended to use this bash script to execute your script, e.g.:

`bash recipes/megatron/bootstrap.sh recipes/megatron/speech_e2e/train_tokenizer.sh`
