# MuLan: Joint Embedding of Music audio and Language

## Description

This task aims to build a joint embedding model for music audio and language. It contains:

- `models/*.py`, which defines models based on AST and BERT;
- `modules/pl_datamodule.py`, which specifies the dataset and dataloading logic;
- `modules/pl_module.py`, which configures the optimizer and scheduler used for training, and describes the logic for training/validation/testing;
- `dataset/wds.py`, which defines webdatalist for all training datasets;
- `dataset/utils.py`, which defines some helper functions for all training datasets;
- `dataset/val.py`, which defines the dataset for validation;
- `dataset/infer.py`, which defines the dataset for inferencing;
- `dataset/*.py`, the remaining files in dataset define all training datasets;
- `dataset/packing/`, which contains the scripts to prepare the datasets;
- `conf/default.yaml`, which specifies the training process;
- `conf/infer.yaml`, which specifies the inferencing process;

## Data

There are multiple audio-text pair datasets used in this project. The data is stored on HDFS with the webdataset format.

Open Source Datasets:

- AudioSet: 1.95M 10s clips, text: intrument tags
- ECALS: 698k 30s clips, text: title, author, tags, etc.
- Karaoke: 698k 30s clips, text: instrument tags

Private Datasets:

- Playlist: 17.15M 30s clips from resso playlist, text: playlist description
- MCC GPT_Gen: 30s clips from MCC 15m dataset, text: gpt generated caption
- MCC N2M: 30s clips from MCC 15m dataset, text: n2m queried captions
- TT Query: 30s clips from tt query data, text: querys
- Short Form: ~108M 30s clips from tt-music, text: title, author, genre, theme, mood, etc.
- Long Form: 30s clips from tt-music, text: title, country

## Run Training

```bash
bash recipes/mulan/bootstrap.sh fit --config recipes/mulan/conf/default.yaml
```

We provide two parameters to specify the dataset to use and the weights of sampling for each dataset.

```yaml
selected_dataset_names: [audio_set, ecals, karaoke]
weights: [0.2, 0.2, 0.2]
```

## Run Inferencing

We use a meta txt as input for inferencing. The meta txt should contain the following fields:

`<audio_npy_filepath>|<extra_fields>...`

Only the audio npy path is required. The extra fields won't be used in the inferencing process.

```bash
bash recipes/mulan/bootstrap.sh predict --config recipes/mulan/conf/infer.yaml --predict_dataset.meta_path=<meta_txt_path>
```

Since we are using ddp, the outputs will be saved individually on each device, and we need to merge them together.
Modify `merge_infer.py` to specify the path to the outputs and use the script to merge them.

## Benchmarks

The benchmarks are conducted on 8 nodes with 8 V100 GPUs.

### Model Benchmark

| | Train Loss | Val_self_median_rank | Val_self_hit_score | Val_kaggle_median_rank | Val_kaggle_hit_score | Notes |
|-|------|------|------|------|-----|----|
| mtpretrain | 4.8 |  | |  | | origin version |
| 20230307 | 4.6 |  | |  | | migrated version |
| 20230327 | - |  | |  | | fp32, merged webdataset, mak dataset |

### Resource Benchmark

| | GPUs | GPU Memory | CPU (Peak/Avg) | RAM | Step Time | Steps |
|-|------|------|------|------|------|-----|
| mtpretrain | 1*8 V100 | 31.3 | 76/25 | 258 | 4.4 s/it | 500 |
| 20230307 | 1*8 V100 | 31.3 | 74/22 | 177 | 3.9 s/it | 500 |
| 20230327 | 1*8 V100 | 26.4 | 32/10 | 142 | 4.7 s/it | 500 |
