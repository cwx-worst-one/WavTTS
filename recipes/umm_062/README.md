# Universal Music Model (UMM)


## Training

### BestRQ Stage 1
1. Prepare datasets (use recipes/datasets/)
2. Calculate zero-mean unit-variance using `python3 recipes/umm/prepare_data_stats.py`
3. Run any of the following for Stage 1 training:
- `bash launch.sh fit --conf ./recipes/umm/conf/stage1_speech.yaml`
- `bash launch.sh fit --conf ./recipes/umm/conf/stage1_music.yaml`

### SoundStorm (Stage 2/3)

**Training**  
Arnold Trial: https://arnold.byted.org/task/4504899

The following config is used for SoundStorm training: `./recipes/umm/conf/soundstorm2_speech.yaml`
Dependent models are defined within the config, respectively:
- audio_model: `!new:recipes.soundstorm2.lightning.soundstream.SoundStreamSpeech24k`
- semantic_model: `!new:recipes.soundstorm2.lightning.bestrq.BestRQMelCTCModel`

You can change the `recipes.soundstorm2.lightning.bestrq.BestRQMelCTCModel` to a new LightningModule, containing inference code to that model.
Same goes for the audio model.

You can run the following command to start training using a BestRQ model checkpoint.

```bash
bash launch.sh fit --conf ./recipes/umm/conf/soundstorm2_speech.yaml
```

**Checkpoints**  
Checkpoints are saved for each trial under the `$ARNOLD_OUTPUT` environment variable. You can check the trial's `stdout` for that path.
Simply copy over the checkpoint to the local filesystem, and run inference as outlined below.


**Inference**  
```bash

# With an A100 GPU, inspect/run this file:
python3 ./recipes/umm/inference/semantic2audio.py
```


Alternatively, there are a few more notebook that go more in-depth for sanity checking the data, model, etc.:
- `recipes/umm/notebooks/umm_soundstorm2.ipynb`



## Datasets
You can download the following datasets for pre-training a speech model:
```bash
bash ./recipes/datasets/libritts/download_libritts.sh
bash ./recipes/datasets/librispeech/download_librispeech.sh
bash ./recipes/datasets/librilight/download_librilight.sh

python3 ./recipes/datasets/libritts/prepare_libritts.py
python3 ./recipes/datasets/librispeech/prepare_librispeech.py
python3 ./recipes/datasets/librilight/prepare_librilight.py
```

### Data Transfer

**US -> Singapore**
```bash
export CLUSTER_NAME=macaw
export QUEUE_NAME=root.macaw_sami_batch
export JOB_NAME=distcp_job_us_to_sg_speech_librilight_large

hadoop distcp \
-Dyarn.cluster.name=$CLUSTER_NAME \
-Dmapreduce.job.queuename=$QUEUE_NAME \
-Dmapreduce.job.name=$JOB_NAME \
-Dmapreduce.map.memory.mb=4000 \
-Dmapreduce.map.cpu.vcores=1 \
-Dmapreduce.reduce.memory.mb=4000  \
-Dmapreduce.reduce.cpu.vcores=1 \
-Dmapreduce.input.fileinputformat.list-status.num-threads=1 \
-skipcrccheck \
-update \
-strategy dynamic \
-bandwidth=50 \
-m 200 \
hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200/ \
hdfs://harunasg/home/byte_speech_sv/data/music/billboard_hot200
```

**Singapore -> China**
```bash
export CLUSTER_NAME=hakes
export QUEUE_NAME=root.hl_lab_audio
export JOB_NAME=distcp_job_sg_to_cn_speech

hadoop distcp \
-Dyarn.cluster.name=$CLUSTER_NAME \
-Dmapreduce.job.queuename=$QUEUE_NAME \
-Dmapreduce.job.name=$JOB_NAME \
-Dmapreduce.map.memory.mb=4000 \
-Dmapreduce.map.cpu.vcores=1 \
-Dmapreduce.reduce.memory.mb=4000  \
-Dmapreduce.reduce.cpu.vcores=1 \
-Dmapreduce.input.fileinputformat.list-status.num-threads=1 \
-skipcrccheck \
-update \
-strategy dynamic \
-bandwidth=50 \
-m 200 \
hdfs://harunasg/home/byte_speech_sv/data/music/billboard_hot200 \
hdfs://haruna/home/byte_speech_sv/data/music/billboard_hot200
```

## Inference
```bash
bash launch.sh predict --conf ./recipes/umm/conf/inference_speech.yaml --trainer.limit_predict_batches 100000
```