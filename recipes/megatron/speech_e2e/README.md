# SAMI Speech LLM Example

## Setup

Please follow `recipes/megatron/README.md`.

## Process data

Run the following commands to download data and process it with your own tokenizer.

```bash
bash recipes/megatron/bootstrap.sh recipes/megatron/speech_e2e/data/prepare_data.sh
bash recipes/megatron/bootstrap.sh recipes/megatron/speech_e2e/train_tokenizer.sh
bash recipes/megatron/bootstrap.sh recipes/megatron/speech_e2e/process_data.sh
```

## Run training

```bash
bash recipes/megatron/bootstrap.sh recipes/megatron/speech_e2e/run.sh

# (optional) In a separate terminal, monitor your GPU memory
watch -n 0.5 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

Some hyperparameters can be overrided by environment variables:

- MICRO_BATCH_SIZE:1
- GLOBAL_BATCH_SIZE:128
- TP_SIZE:2
- PP_SIZE:2
- SEQ_PARALLEL:1
- DIST_OPTIMIZER:1
- FLASH_ATTN:1
- ACTIVATION_CKPT:1

- NLAYERS:30
- NHIDDEN:4096
- NHEADS:32
- SEQ_LEN:2048
