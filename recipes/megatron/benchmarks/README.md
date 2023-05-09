# SAMI Benchmarks for Megatron LM

## Setup

Please follow `recipes/megatron/README.md`.

## Process data

There are 2 options to prepare your data for benchmarks.

### Option 1

Run the following script to download dataset in jsonl format and process it with your own tokenizer.

```bash
bash recipes/megatron/bootstrap.sh recipes/megatron/benchmarks/process_data.sh
```

### Option 2

Run the following commands to download dataset in binary format. You won't be able to use your own tokenizer with this option.

```bash
hdfs dfs -get \
    /home/byte_speech_sv/jingsong.gao/sami_ai_llm/data/openwebtext_text_document.* \
    recipes/megatron/benchmarks/data/
```

## Run benchmark

```bash
bash recipes/megatron/bootstrap.sh recipes/megatron/benchmarks/run.sh

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
