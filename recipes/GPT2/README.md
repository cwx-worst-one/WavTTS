# GPT2&3 with Lightning & DeepSpeed

## Training Billion+ Parameter GPT Models

This example shows how to train large GPT models (with huggingface's GPT definition). You can train GPT models with different size of parameters using a single V100 GPU and amount of CPU/Mem resources. To speedup the training, you can use deepspeed strategies, e.g., `deepspeed_stage_3` or `deepspeed_stage_3_offload`. To further enable large batch size, you can set `enable_activation_ckpt` to be `True` in order to utilize gradient checkpointing technique. By default, this flag is on.

A list of GPT models are tested based on the setup specified in `conf/default.yaml`, with `fp16` being enabled during training.

### gpt2_tiny (7.4M Params, with 32CPU/64GiB Mem/1 * V100)

```bash
python3 -m samantha.main fit --config recipes/GPT2/conf/default.yaml --training_params.model gpt2_tiny --training_params.batch_size 32 --run_opts.strategy deepspeed_stage_3
```

### gpt2_small (124M Params, with 32CPU/64GiB Mem/1 * V100)
```bash
python3 -m samantha.main fit --config recipes/GPT2/conf/default.yaml --training_params.model gpt2_small --training_params.batch_size 32 --run_opts.strategy deepspeed_stage_3
```

### gpt2_medium (354M Params, with 32CPU/64GiB Mem/1 * V100)

```bash
python3 -m samantha.main fit --config recipes/GPT2/conf/default.yaml --training_params.model gpt2_medium --training_params.batch_size 32 --run_opts.strategy deepspeed_stage_3
```

### gpt2_large (774M Params, with 32CPU/64GiB Mem/1 * V100)

```bash
python3 -m samantha.main fit --config recipes/GPT2/conf/default.yaml --training_params.model gpt2_large --training_params.batch_size 32 --run_opts.strategy deepspeed_stage_3_offload
```

### gpt2_xl (1.6B Params, with 32CPU/64GiB Mem/1 * V100)

```bash
python3 -m samantha.main fit --config recipes/GPT2/conf/default.yaml --training_params.model gpt2_xl --training_params.batch_size 32 --run_opts.strategy deepspeed_stage_3_offload
```

### gpt2_2B (2.1B Params, with 32CPU/64GiB Mem/1 * V100)
```bash
python3 -m samantha.main fit --config recipes/GPT2/conf/default.yaml --training_params.model gpt2_2B --training_params.batch_size 32 --run_opts.strategy deepspeed_stage_3_offload
```

### gpt2_3B (3.2B Params, with 32CPU/128GiB Mem/1 * V100)
```bash
python3 -m samantha.main fit --config recipes/GPT2/conf/default.yaml --training_params.model gpt2_3B --training_params.batch_size 32 --run_opts.strategy deepspeed_stage_3_offload
```

### gpt2_4B (4.2B Params, with 32CPU/128GiB Mem/1 * V100)
```bash
python3 -m samantha.main fit --config recipes/GPT2/conf/default.yaml --training_params.model gpt2_4B --training_params.batch_size 16 --run_opts.strategy deepspeed_stage_3_offload
```

### gpt2_pretrained (124M Params, with 32CPU/128GiB Mem/1 * V100)
```bash
python3 -m samantha.main fit --config recipes/GPT2/conf/default.yaml --training_params.model gpt2_pretrained --training_params.batch_size 32 --run_opts.strategy deepspeed_stage_3_offload
```

## Benchmarks

This is a demo project for GPT training, no model benchmark is needed.

### Resource Benchmark

Huggingface vs Panther

> Environment
> - 1 A100 GPU
> - icm: samantha, torch1.10, python3.7


1. gpt2_2B with strategy `ddp_find_unused_parameters_false` and `deepspeed_stage_3_offload`

```shell
python3 -m samantha.main fit \
  --config recipes/GPT2/conf/default.yaml \
  --training_params.model gpt2_2B \
  --training_params.batch_size 16 \
  --training_params.provider <provider> \
  --run_opts.precision=16 \
  --run_opts.strategy=<strategy>
```

|  Provider   | batch size |             strategy             | GPU Memory(GB) | Step Time (s/it) | Speed Up |
|:-----------:|:----------:|:--------------------------------:|:--------------:|:----------------:|:--------:|
| huggingface |     16     | ddp_find_unused_parameters_false |      66.8      |       3.30       |          |
|   panther   |     16     | ddp_find_unused_parameters_false |      64.0      |       2.24       |   47%    |
| huggingface |     16     |    deepspeed_stage_3_offload     |      19.3      |       6.36       |          |
|   panther   |     16     |    deepspeed_stage_3_offload     |      24.8      |       5.79       |   9.8%   |

2. gpt2_8B with strategy `deepspeed_stage_3_offload`

```shell
python3 -m samantha.main fit \
  --config recipes/GPT2/conf/default.yaml \
  --training_params.model gpt2_8B \
  --training_params.batch_size 16 \
  --training_params.provider <provider> \
  --run_opts.precision=16 \
  --run_opts.strategy=deepspeed_stage_3_offload
```

|  Provider   | batch size | GPU Memory(GB) | Step Time (s/it) | Speed Up |
|:-----------:|:----------:|:--------------:|:----------------:|:--------:|
| huggingface |     16     |      26.3      |      26.45       |          |
|   panther   |     16     |      37.1      |      22.41       |   18%    |
