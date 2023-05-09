# GPT2&3 with Cruise

## Training Billion+ Parameter GPT Models

This example shows how to train large GPT models using [Cruise](https://bytedance.feishu.cn/wiki/wikcndoXvN2g2tuQSF76mebqnwh), our internal training framework provided by AML team. You can train GPT models with different size of parameters using a single V100 GPU and amount of CPU/Mem resources.

A list of GPT models are tested based on the setup specified in `conf/default.yaml`, with `fp16` being enabled during training.

### install requirements

```bash
pip3 install -r recipes/cruise_GPT2/requirements.txt
```

### gpt2_tiny (7.4M Params)

To speedup the training, we suggest using `deepspeed` strategy provided by Cruise.

```bash
python3 -m samantha.main fit --config recipes/cruise_GPT2/conf/deepspeed.yaml --training_params.model gpt2_tiny
```

Cruise also offers `fsdp` strategy.

```bash
TORCHRUN -m samantha.main fit --config recipes/cruise_GPT2/conf/fsdp.yaml --training_params.model gpt2_tiny
```

Another way to create training tasks using Cruise is to use CruiseCLI, as shown in the following command:

```bash
python3 recipes/cruise_GPT2/run.py --trainer=recipes/cruise_GPT2/conf/cruise_ds.yaml --model.model_name=gpt2_tiny
```
