# minGPT with Lightning & DeepSpeed

**Lightning now has their own Lightning GPT Example! Highly recommend using their repo [here](https://github.com/Lightning-AI/lightning-GPT).**

*Note: this minimal example won't be as efficient/optimized as other specialized repos due to keeping it minimal and readable, but large model training is still achievable.* 

## Training Billion+ Parameter GPT Models

A lot of information has been taken from the very helpful [Lightning Model Parallel Documentation](https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#fully-sharded-training).

In the below examples batch size is set to 1 to try reduce VRAM as much as possible, but you can scale that with your compute. In the below case we could scale the batch size significantly to fill the left over GPU memory. You can try to set `batch_size` to 512 on V100.

For 20B/45B parameter models, you'll need a reasonable amount of CPU RAM as we offload partitions to the CPU. For the 45B parameter model, you'll need around 1TB of CPU memory.

Note that we enable CPU offloading. Offloading has a huge impact on throughput and in most cases when training from scratch should be turned off. You should consider scaling the number of GPUs rather than enabling offloading at these model sizes.

##### 1.7B (Requires around 2GiB per 8 GPUs, 5.1GiB for 1 GPU)
```bash
python3 -m samantha.main fit --config recipes/minGPT/conf/default.yaml
```

##### ~10B (Requires around 6GiB per 8 GPUs, 26GiB for 1 GPU)
```bash
python3 -m samantha.main fit --config recipes/minGPT/conf/default.yaml --training_params.n_layer 13 --training_params.n_head 16 --training_params.n_embd 8192
```

##### ~20B (Requires around 8GiB per 8 GPUs, OOM for 1 GPU, offloading onto ~500GB of CPU RAM)
```bash
python3 -m samantha.main fit --config recipes/minGPT/conf/default.yaml --training_params.n_layer 25 --training_params.n_head 16 --training_params.n_embd 8192
```

##### ~45B (Requires around 14GiB per 8 GPUs, OOM for 1 GPU, offloading onto ~950GB of CPU RAM)
```bash
python3 -m samantha.main fit --config recipes/minGPT/conf/default.yaml --training_params.n_layer 56 --training_params.n_head 16 --training_params.n_embd 8192
```
