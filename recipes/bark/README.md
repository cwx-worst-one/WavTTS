- [AudioLM](#audio-lm)
  - [Quick Start](#quick-start)
  - [Migration Detail](#migration-detail)
    - [Configuration](#conf)
    - [Modules](#modules)
    - [Lightning Modules](#lightning-modules)
    - [Datasets](#datasets)
    - [Requires](#requires)
  - [End-to-End Benchmark](#end-to-end-benchmark)
    - [@A100](#a100-1gpu-80gb)
    - [@V100](#v100-1gpu-32gb)

# Audio LM

## Quick Start
For coarse model (same with fine, semantic model training, just replace config yaml).
It may cost several minutes to install `flash attn` when you first run commands below.
If you don't want it, feel free to comment out corresponding lines in [bootstrap.sh](./bootstrap.sh).

**note**: work-directory should be `sami_ai_models`
```shell
# training with original huggingface
bash recipes/audio_lm/bootstrap.sh fit --config recipes/audio_lm/conf/coarse.yaml --trainer.benchmark=True

# with fp16 precision
bash recipes/audio_lm/bootstrap.sh fit --config recipes/audio_lm/conf/coarse.yaml --run_opts.precision=16

# with activation checkpointing
bash recipes/audio_lm/bootstrap.sh fit --config recipes/audio_lm/conf/coarse.yaml --run_opts.checkpointing=True

# with flash attention
bash recipes/audio_lm/bootstrap.sh fit --config recipes/audio_lm/conf/coarse.yaml --run_opts.flash_attn=flash_attn

# with flash attention cuda (cuda ext must @A100 and with fp16 precision)
bash recipes/audio_lm/bootstrap.sh fit --config recipes/audio_lm/conf/coarse.yaml --run_opts.flash_attn=flash_attn_cuda --run_opts.precision=16
```

## Migration Detail
All training needed files are migrated to this recipe from [bytegen](https://code.byted.org/litang.frank/bytegen/tree/colin/dev5-migrate).
```text
recipes/audio_lm
├── README.md
├── bootstrap.sh
├── conf
├── datasets
├── modules
├── lit_modules
├── requires
└── utils
```

### Conf
```text
recipes/audio_lm/conf
├── coarse.yaml
├── fine.yaml
├── hparams.yaml
├── semantic.yaml
└── semantic_wds.yaml
```
For coarse, fine and semantic yaml file, they are corresponding a specific training task.
`hparams.yaml` is a copy from `bytegen` repo, and all previous three yaml file `include` it.
`semantic_wds.yaml`is semantic model with webdataset.

We use `Hyperpyyaml` as our repo configuration, which we can initialize everything like
`dataset`, `dataloader`, `model`, etc. in it without any python code. For more syntax usage,
please refer to [README](../../README.md#yaml-config)

### Modules
```text
recipes/audio_lm/modules
├── __init__.py
├── gpt.py
└── loss.py
```
`gpt.py` defined the language model which already wrapper flash attention and huggingface, user can
just pass `--run_opts.flash_attn=flash_attn` or `--run_opts.flash_attn=null` to use **flash attn** or not,
don't have to pay more attention about the model detail.

`loss.py` defined masked cross_entropy loss.

### Lightning Modules
```text
recipes/audio_lm/lit_modules
├── __init__.py
├── lit_data.py
├── lit_coarse.py
├── lit_fine.py
└── lit_semantic.py
```
Except `datamodule`, each `lit_xx` defined the same forward processing, which called `training_step` here, as `bytegen` repo.

As each model has on-the-fly dependencies, we defined a method called `load_required_modules` which will be called at setup phase.

And corresponding required modules, you may find its ckpt path and initialization method at yaml file.
```yaml
    required_modules:
        mulan:
            - hdfs://haruna/home/byte_speech_sv/user/litang/mulan/2023-02-24_audio_emb_asset4/mulan_2023-02-24_v3.pt
            - !name:recipes.audio_lm.requires.model_initializer.init_mulan
              cache_dir: !ref <run_opts[cache_dir]>
        w2v:
            - hdfs://haruna/home/byte_speech_sv/user/litang/mulan/2023-02-24_audio_emb_asset4/
            - !name:recipes.audio_lm.requires.model_initializer.init_w2v
              cache_dir: !ref <run_opts[cache_dir]>
        sound_stream:
            - hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/pretrained/sound_stream_1000k/
            - !name:recipes.audio_lm.requires.model_initializer.init_sound_stream
              cache_dir: !ref <run_opts[cache_dir]>
```

### Datasets
```text
recipes/audio_lm/datasets
├── __init__.py
├── dataset.py
├── wds.py
└── wds_packer.py
```
`dataset.py` defined dataset which same as `bytegen` repo.

`wds.py` refactored original dataset into `webdataset`.

`wds_packer.py` defined packing method.

### Requires
```text
recipes/audio_lm/requires
├── model_initializer.py
├── mulan
└── w2v
```
`model_initializer.py` defined how to load each required model mentioned above.

`mulan` and `w2v` are same as `bytegen`.

## End-to-End Benchmark
This is just a part of whole report, for more detail please refer to [GPT Benchmark](https://bytedance.feishu.cn/docx/WO4SdnnWEoSY6txnMUZcZJqtn7f#YKaWdCaYyoqwkQxc1YNcZFcfncd),

common setup
- sequence length: 5553
- nlayer-nhead: 18-8
- checkpointing: Yes

**note**: under cuda11.3 and pytorch 1.10, HuggingFace with fp16 is slower than fp32 @A100.

### @A100, 1GPU, 80GB

|               | strategy | precision | batch size | gpu mem   | speed (+-0.02) | speed-up |
|---------------|----------|-----------|------------|-----------|----------------|----------|
| HuggingFace   | ddp      | fp32      | 8          | 56_624MiB | 7.24  s/it     |          |
| FlashAttnCuda | ddp      | fp16      | 8          | 20_108MiB | 2.17  s/it     | 3.34     |

### @V100, 1GPU, 32GB

|             | strategy | precision | batch size | gpu mem   | speed (+-0.02) | speed-up |
|-------------|----------|-----------|------------|-----------|----------------|----------|
| HuggingFace | ddp      | fp32      | 2          | 20_188MiB | 5.9  s/it      |          |
| FlashAttn   | ddp      | fp16      | 2          | 15_456MiB | 2.5  s/it      | 2.36     |
