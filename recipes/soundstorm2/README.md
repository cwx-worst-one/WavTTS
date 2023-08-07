# SoundStorm
PyTorch implementation of "SoundStorm: Efficient Parallel Audio Generation" by Z. Borsos et al. (2023)

Paper: https://arxiv.org/pdf/2305.09636.pdf

We present SoundStorm, a model for efficient, non-autoregressive audio generation. SoundStorm receives as input the semantic tokens of AudioLM, and relies on bidirectional attention and confidence-based parallel decoding to generate the tokens of a neural audio codec. Compared to the autoregressive generation approach of AudioLM, our model produces audio of the same quality and with higher consistency in voice and acoustic conditions, while being two orders of magnitude faster. SoundStorm generates 30 seconds of audio in 0.5 seconds on a TPU-v4. We demonstrate the ability of our model to scale audio generation to longer sequences by synthesizing high-quality, natural dialogue segments, given a transcript annotated with speaker turns and a short prompt with the speakers’ voices. Audio samples are available at https://google-research.github. io/seanet/soundstorm/examples/


## Quickstart
The following command will train a SoundStorm model on 30s audio samples from the Karaoke dataset, using:
1. SoundStream v2 (12 quantizers, 1024 codebook size, 50Hz)
2. w2v-Conformer (1024 codebook size, 25Hz)

```bash
bash launch.sh fit --conf ./recipes/soundstorm/conf/default.yaml
```

## Gradio Server  
You can run an inference server as follows. This will download and load the most recent checkpoint from Arnold, and allows you to interactively generate music at the speed of light ⚡️:
```bash
python3 ./recipes/soundstorm/inference/gradio_server.py
```

Alternatively, you can also run regular inference scripts:
```bash
python3 ./recipes/soundstorm/inference/semantic2audio.py [local_ckpt_path]
```

### Create a test dataset
```bash
python3 ./recipes/soundstorm/tests/create_test_dataset.py

```

## Todo
- [x] Time-align semantic tokens with audio frame rate
- [x] Interleave time-aligned semantic/audio tokens
- [x] Test time-alignment
- [x] Test interleaving
- [x] Masking scheme
- [x] Test masking
- [x] Conformer:
  - [x] Conformer base architecture
  - [x] (Bidirectional) attention
  - [ ] Test Conformer
- [x] Confidence-based parallel decoding
- [x] Test confidence-based parallel decoding

### Optional:
- [ ] Voice preservation
- [ ] Acoustic consistency drift


# Models

## Transfer Models

### US -> China

**US -> Singapore**
```bash
CLUSTER_NAME=macaw QUEUE_NAME=root.macaw_sami_batch JOB_NAME=distcp_job_us_to_sg_speech_librilight_large hadoop distcp \
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
hdfs://harunava/home/byte_speech_sv/models/* \
hdfs://harunasg/home/byte_speech_sv/models
```

**Singapore -> China**
```bash
CLUSTER_NAME=hakes QUEUE_NAME=root.hl_lab_audio JOB_NAME=distcp_job_sg_to_cn hadoop distcp \
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
hdfs://harunasg/home/byte_speech_sv/models/* \
hdfs://haruna/home/byte_speech_sv/models
```

### China -> US

**China -> Singapore**
```bash
export CLUSTER_NAME=hakes
export QUEUE_NAME=root.hl_lab_audio
export JOB_NAME=distcp_job_cn_to_sg

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
hdfs://haruna/home/byte_speech_sv/models/* \
hdfs://harunasg/home/byte_speech_sv/models
```

**Singapore -> US**
```bash
CLUSTER_NAME=macaw QUEUE_NAME=root.macaw_sami_batch JOB_NAME=distcp_job_us_to_sg hadoop distcp \
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
hdfs://harunasg/home/byte_speech_sv/models/* \
hdfs://harunava/home/byte_speech_sv/models
```
