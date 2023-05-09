# ByteFormers example
This is a minimal example of using optimised Transformer base components to train a GPT-2 model on the Shakespeare dataset for character-level language modeling. It should achieve around 145 TFlops on a single A100 GPU (which is around the same reported in the Megatron work).

Under the hood, it uses:
1. [Triton](https://github.com/openai/triton)
2. [Fully-Sharded Data Parallel](https://engineering.fb.com/2021/07/15/open-source/fsdp/)
3. [Lightning Fabric](https://lightning.ai/pages/open-source/fabric/) with a custom trainer written for scaling (`byteformers.trainer.FabricTrainer`)


## Quickstart
1. Get the dataset:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
2. Install requirements:
```bash
pip3 install -r ./recipes/byteformers_example/requirements.txt
```
3. Run model:
```bash
python3 ./recipes/byteformers_example/main.py 
```

## Model weights
Meta's originally published LLaMA model weights (7 - 65 billion) and the SentencePiece tokenizer are currently stored on HDFS:
```
hdfs dfs -ls hdfs://harunava/home/byte_speech_sv/models/llama
```

## RedPajama Dataset
We can train/fine-tune a language model model on the (reproduced) dataset that the original LLaMA 7B-65B models were trained on.
This is a dataset consisting of ~1.2 trillion language tokens, recreated in an effort led by [Together](https://www.together.xyz/blog/redpajama).

The following datasets are included:
|               |   RedPajama   |   LLaMA*      |
|---------------|---------------|---------------|
| CommonCrawl   | 878 billion   | 852 billion   |
| C4            | 175 billion   | 190 billion   |
| Github        | 59 billion    | 100 billion   |
| Books         | 26 billion    | 25 billion    |
| ArXiv         | 28 billion    | 33 billion    |
| Wikipedia     | 24 billion    | 25 billion    |
| StackExchange | 20 billion    | 27 billion    |


A (multiprocessed) download script that also verifies the sha256 hashes can be launched as follows:

```bash
python3 ./recipes/byteformers_example/scripts/download_redpajama.py
```

The dataset is already accesssible at:
```bash
hdfs dfs -ls hdfs://harunava/home/byte_speech_sv/data/redpajama
```