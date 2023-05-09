# Audio Diffusion

## Quickstart
```
mkdir ./data
hdfs dfs -copyToLocal hdfs://harunava/home/byte_speech_sv/data/resso_1.3m_webdataset/resso_0_2 ./data

```

## Description


## Logging
We currently log all experiments inside `$ARNOLD_OUTPUT` by default. You can run a TensorBoard instance with:
```
tensorboard --logdir $ARNOLD_OUTPUT --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all
```

To show previous trials, you could run:
```
tensorboard --logdir hdfs://harunava/home/byte_speech_sv/audio_diffusion/tasks --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all
```

## Running tests
Tests can be run using the following command:
```
pytest -sv ./recipes/audio_diffusion/tests/
```

Some tests will be skipped, as they are only meant to be run on Merlin (with a GPU and access to HDFS)
