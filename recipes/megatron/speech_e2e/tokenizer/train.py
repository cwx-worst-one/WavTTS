import argparse
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from recipes.megatron.tools.dataset.tokenizer import train_tokenizer


def tokenizer_and_trainer_provider():
    audio_tokens = [f"<audio{i}>" for i in range(1000)] + ["[EOS]", "[SEP]", "[PAD]"]
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=audio_tokens,
        continuing_subword_prefix="@",
    )
    return tokenizer, trainer


def tf_special_tokens_provider():
    return {"eos_token": "[EOS]", "sep_token": "[SEP]", "pad_token": "[PAD]"}


def train_data_iterator():
    batch = []
    batch_size = args.batch_size
    with open(args.input_path, "r") as fin:
        for line in fin.readlines():
            line = json.loads(line)
            if not isinstance(line, dict):
                line = json.loads(line)
            if len(batch) == batch_size:
                yield batch
                batch = []
            batch.append(line["targets"])
    if len(batch) > 0:
        yield batch


def test_data_iterator():
    with open(args.input_path, "r") as fin:
        for line in fin.readlines():
            line = json.loads(line)
            if not isinstance(line, dict):
                line = json.loads(line)
            yield line["inputs"] + " [SEP] " + line["targets"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path to input jsonl file")
    parser.add_argument("output_path", type=str, help="Path to save tokenizer")
    parser.add_argument("--vocab_size", type=int, default=2000, help="Vocab size")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    args = parser.parse_args()

    tf_tokenizer = train_tokenizer(
        tokenizer_and_trainer_provider,
        train_data_iterator,
        test_data_iterator,
        tf_special_tokens_provider,
        output_path=args.output_path,
    )
