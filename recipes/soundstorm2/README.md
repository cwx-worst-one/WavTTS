# SoundStorm
PyTorch implementation of "SoundStorm: Efficient Parallel Audio Generation" by Z. Borsos et al. (2023)

Paper: https://arxiv.org/pdf/2305.09636.pdf

We present SoundStorm, a model for efficient, non-autoregressive audio generation. SoundStorm receives as input the semantic tokens of AudioLM, and relies on bidirectional attention and confidence-based parallel decoding to generate the tokens of a neural audio codec. Compared to the autoregressive generation approach of AudioLM, our model produces audio of the same quality and with higher consistency in voice and acoustic conditions, while being two orders of magnitude faster. SoundStorm generates 30 seconds of audio in 0.5 seconds on a TPU-v4. We demonstrate the ability of our model to scale audio generation to longer sequences by synthesizing high-quality, natural dialogue segments, given a transcript annotated with speaker turns and a short prompt with the speakers’ voices. Audio samples are available at https://google-research.github. io/seanet/soundstorm/examples/


## Quickstart

### Speech
The following command will train a SoundStorm model on 24kHz, 2s - 30s audio samples from the LibriTTS dataset, using:
1. SoundStream (12 quantizers, 1024 codebook size, 50Hz)
2. BestRQ, fine-tuned with mel spectrogram reconstruction and CTC loss

```bash
bash launch.sh fit --conf ./recipes/soundstorm/conf/speech/libritts.yaml
```

### Music
The following command will train a SoundStorm model on 44.1kHz, 30s audio samples from the Billboard 200 Dataset, using:
1. DAC (9 quantizers, 1024 codebook size, approx. 87Hz)

```bash
bash launch.sh fit --conf ./recipes/soundstorm/conf/music/billboard.yaml
```

## Inference
Jupyter Notebook: You can use the `./recipes/soundstorm2/notebooks/soundstorm inference.ipynb` as reference.
You can also use the inference scripts to use a pre-trained SoundStorm checkpoint for inference:
```bash
python3 ./recipes/soundstorm2/inference/audio.py [CKPT_PATH]
```

Alternatively, a Gradio server can be launched.


### Gradio Server  
The Gradio server allows you to interactively generate music at the speed of light ⚡️:
```bash
python3 ./recipes/soundstorm2/inference/gradio_server.py CKPT_PATH
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
[TODO] Add models to HDFS


**US -> Singapore**
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
hdfs://harunasg/home/byte_speech_sv/data/music/billboard_hot200_mp3/ \
hdfs://haruna/home/byte_speech_sv/data/music/billboard_hot200_mp3/
